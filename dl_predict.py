"""
今彩539 深度學習預測模組
模型：LSTM + Attention（注意力機制）
原理：自動學習哪幾期歷史對預測最重要
注意：torch 採 lazy import，避免 macOS Streamlit fork crash
"""

from __future__ import annotations
import hashlib
from pathlib import Path
from analyzer import Draw

# ── torch 在函數內 lazy import，避免 macOS fork crash ──
def _torch_available() -> bool:
    try:
        import torch  # noqa
        return True
    except ImportError:
        return False

DL_AVAILABLE = _torch_available()

SEQ_LEN     = 30
ALL_NUMS    = list(range(1, 40))
CACHE_DIR   = Path(__file__).parent / "model_cache"
EPOCHS      = 80
BATCH_SIZE  = 64
LR          = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3


def _lstm_hash(draws: list[Draw]) -> str:
    key = f"lstm_{len(draws)}_{draws[-1].period}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _build_model():
    """在函數內 import torch，避免模組載入時 crash"""
    import torch
    import torch.nn as nn

    class LotteryLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=39, hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS, batch_first=True,
                dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
            )
            self.attention = nn.Linear(HIDDEN_SIZE, 1)
            self.dropout   = nn.Dropout(DROPOUT)
            self.fc        = nn.Linear(HIDDEN_SIZE, 39)

        def forward(self, x):
            lstm_out, _  = self.lstm(x)
            attn_scores  = self.attention(lstm_out)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context      = (attn_weights * lstm_out).sum(dim=1)
            context      = self.dropout(context)
            out          = torch.sigmoid(self.fc(context))
            return out, attn_weights.squeeze(-1)

    return LotteryLSTM()


def _build_tensors(draws: list[Draw]):
    import torch
    X, y = [], []
    for i in range(SEQ_LEN, len(draws) - 1):
        seq    = [[1.0 if n in draws[j].numbers else 0.0 for n in ALL_NUMS]
                  for j in range(i - SEQ_LEN, i)]
        target = [1.0 if n in draws[i].numbers else 0.0 for n in ALL_NUMS]
        X.append(seq)
        y.append(target)
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))


def train_lstm(
    draws: list[Draw],
    progress_cb=None,
    use_cache: bool = True,
) -> dict:
    if not DL_AVAILABLE:
        return {"probs": {n: 0.0 for n in ALL_NUMS}, "attn": [], "from_cache": False}

    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn

    # macOS 防 crash
    torch.set_num_threads(1)

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"lstm_{_lstm_hash(draws)}.pt"

    if use_cache and cache_file.exists():
        data = torch.load(cache_file, map_location="cpu", weights_only=False)
        return {**data, "from_cache": True}

    X, y = _build_tensors(draws)

    n = len(X)
    sample_w = torch.ones(n)
    sample_w[-min(500, n):] = 2.0

    loader = DataLoader(TensorDataset(X, y, sample_w),
                        batch_size=BATCH_SIZE, shuffle=True)

    model     = _build_model()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCELoss(reduction="none")

    model.train()
    for epoch in range(EPOCHS):
        for xb, yb, wb in loader:
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = (criterion(pred, yb).mean(dim=1) * wb).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        if progress_cb:
            progress_cb(epoch + 1, EPOCHS)

    model.eval()
    with torch.no_grad():
        x_last = torch.tensor([[
            [1.0 if n in draws[i].numbers else 0.0 for n in ALL_NUMS]
            for i in range(len(draws) - SEQ_LEN, len(draws))
        ]], dtype=torch.float32)
        probs, attn = model(x_last)
        probs_np = probs.squeeze(0).numpy()
        attn_np  = attn.squeeze(0).numpy()

    result = {
        "probs": {n: float(probs_np[i]) for i, n in enumerate(ALL_NUMS)},
        "attn":  attn_np.tolist(),
    }

    for f in CACHE_DIR.glob("lstm_*.pt"):
        f.unlink(missing_ok=True)
    torch.save(result, cache_file)

    return {**result, "from_cache": False}


def get_lstm_recommendation(
    draws: list[Draw],
    killed: set[int],
    progress_cb=None,
    use_cache: bool = True,
) -> dict:
    result    = train_lstm(draws, progress_cb, use_cache)
    base_rate = 5 / 39

    remaining = {n: result["probs"][n] for n in ALL_NUMS if n not in killed}
    ranked    = sorted(remaining.items(), key=lambda x: -x[1])

    detail = []
    for n, p in ranked[:10]:
        detail.append({
            "號碼":     f"{n:02d}",
            "LSTM機率": f"{p:.1%}",
            "基準":     f"{base_rate:.1%}",
            "超出基準": f"+{p-base_rate:.1%}" if p > base_rate else f"{p-base_rate:.1%}",
            "建議": "✅ 強烈" if p > base_rate * 1.3 else (
                    "👍 推薦" if p > base_rate * 1.1 else "⚠️ 普通"),
        })

    attn = result.get("attn", [])
    attn_display = []
    if attn and len(draws) >= SEQ_LEN:
        for i, w in enumerate(attn[-SEQ_LEN:]):
            draw_idx = len(draws) - SEQ_LEN + i
            attn_display.append({
                "期數":   draws[draw_idx].period,
                "號碼":   " ".join(f"{n:02d}" for n in draws[draw_idx].numbers),
                "注意力": round(float(w), 4),
            })

    return {
        "top5":         sorted([n for n, _ in ranked[:5]]),
        "top6":         sorted([n for n, _ in ranked[:6]]),
        "top7":         sorted([n for n, _ in ranked[:7]]),
        "all_probs":    result["probs"],
        "attn_display": attn_display,
        "detail":       detail,
        "from_cache":   result["from_cache"],
        "base_rate":    base_rate,
    }
