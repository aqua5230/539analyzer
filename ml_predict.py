"""
今彩539 機器學習預測模組 v2
模型：RandomForest + GradientBoosting + XGBoost 三模型投票
特徵：前5期號碼、遺漏期數、頻率、奇偶、和值、跨度（含近期加權）
快取：模型存到 model_cache/，下次直接載入不用重新訓練
新增：特徵重要性分析（按類別分組）
"""

from __future__ import annotations
import hashlib
import numpy as np
from collections import Counter
from pathlib import Path
from analyzer import Draw

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

LOOKBACK   = 5
ALL_NUMS   = list(range(1, 40))
CACHE_DIR  = Path(__file__).parent / "model_cache"


def _data_hash(draws: list[Draw]) -> str:
    key = f"{len(draws)}_{draws[-1].period}_{draws[-1].numbers}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def get_feature_names() -> list[str]:
    names = []
    for k in range(LOOKBACK):
        names += [f"前{k+1}期_n{n:02d}" for n in ALL_NUMS]
    names += [f"遺漏_{n:02d}" for n in ALL_NUMS]
    names += [f"近30頻率_{n:02d}" for n in ALL_NUMS]
    names += [f"近10頻率_{n:02d}" for n in ALL_NUMS]
    names += ["奇數個數", "大號個數", "和值", "跨度"]
    return names


def build_features(draws: list[Draw], idx: int) -> np.ndarray:
    feats = []
    window = draws[max(0, idx - LOOKBACK + 1): idx + 1]

    # 前N期號碼 one-hot
    for k in range(LOOKBACK):
        d = window[-(k + 1)] if k < len(window) else None
        feats.extend([1 if d and n in d.numbers else 0 for n in ALL_NUMS])

    # 遺漏期數
    last_seen = {}
    for j in range(idx + 1):
        for n in draws[j].numbers:
            last_seen[n] = j
    feats.extend([min(idx - last_seen.get(n, 0), 100) for n in ALL_NUMS])

    # 近30期頻率
    recent30 = draws[max(0, idx - 29): idx + 1]
    freq30 = Counter(x for d in recent30 for x in d.numbers)
    feats.extend([freq30.get(n, 0) for n in ALL_NUMS])

    # 近10期頻率
    recent10 = draws[max(0, idx - 9): idx + 1]
    freq10 = Counter(x for d in recent10 for x in d.numbers)
    feats.extend([freq10.get(n, 0) for n in ALL_NUMS])

    # 當期統計指標
    cur = draws[idx]
    feats += [
        sum(1 for n in cur.numbers if n % 2 == 1),
        sum(1 for n in cur.numbers if n >= 20),
        sum(cur.numbers),
        cur.numbers[-1] - cur.numbers[0],
    ]
    return np.array(feats, dtype=np.float32)


def _train_models(draws: list[Draw], progress_cb=None) -> dict:
    """訓練三個模型，回傳每個號碼的三模型預測機率 + 特徵重要性。"""
    n_samples = len(draws) - 1
    X = np.array([build_features(draws, i) for i in range(n_samples)])

    # 近期樣本加權（近500期權重×2）
    sample_weight = np.ones(n_samples)
    sample_weight[-min(500, n_samples):] = 2.0

    feat_names = get_feature_names()
    importance_sum = np.zeros(len(feat_names))

    results: dict = {}

    for idx, num in enumerate(ALL_NUMS):
        if progress_cb:
            progress_cb(idx + 1, 39)

        y = np.array([1 if num in draws[i + 1].numbers else 0
                      for i in range(n_samples)])

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=15,
            random_state=42, n_jobs=1)
        gb = GradientBoostingClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.05,
            random_state=42)
        xgb = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            eval_metric="logloss",
            random_state=42, verbosity=0, nthread=1)

        rf.fit(X, y, sample_weight=sample_weight)
        gb.fit(X, y, sample_weight=sample_weight)
        xgb.fit(X, y, sample_weight=sample_weight)

        x_pred = build_features(draws, len(draws) - 1).reshape(1, -1)

        def get_prob(m):
            p = m.predict_proba(x_pred)[0]
            return float(p[1]) if len(p) == 2 else float(y.mean())

        p_rf  = get_prob(rf)
        p_gb  = get_prob(gb)
        p_xgb = get_prob(xgb)
        # 加權投票：XGBoost 0.4, GBM 0.35, RF 0.25
        p_vote = p_xgb * 0.4 + p_gb * 0.35 + p_rf * 0.25

        results[num] = {"rf": p_rf, "gb": p_gb, "xgb": p_xgb, "vote": p_vote}
        importance_sum += xgb.feature_importances_

    # 特徵重要性按類別分組
    imp = importance_sum / len(ALL_NUMS)
    n = len(ALL_NUMS)
    results["__importance__"] = {
        "前N期號碼": round(float(imp[:LOOKBACK*n].sum()), 4),
        "遺漏期數":   round(float(imp[LOOKBACK*n:LOOKBACK*n+n].sum()), 4),
        "近30期頻率": round(float(imp[LOOKBACK*n+n:LOOKBACK*n+2*n].sum()), 4),
        "近10期頻率": round(float(imp[LOOKBACK*n+2*n:LOOKBACK*n+3*n].sum()), 4),
        "統計特徵":   round(float(imp[LOOKBACK*n+3*n:].sum()), 4),
    }

    return results


def train_and_predict(
    draws: list[Draw],
    progress_cb=None,
    use_cache: bool = True,
) -> dict:
    if not ML_AVAILABLE:
        return {n: {"rf": 0, "gb": 0, "xgb": 0, "vote": 0} for n in ALL_NUMS}

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"model_{_data_hash(draws)}.pkl"

    if use_cache and cache_file.exists():
        return joblib.load(cache_file)

    results = _train_models(draws, progress_cb)

    if use_cache:
        for f in CACHE_DIR.glob("model_*.pkl"):
            f.unlink(missing_ok=True)
        joblib.dump(results, cache_file)

    return results


def get_ml_recommendation(
    draws: list[Draw],
    killed: set[int],
    progress_cb=None,
    use_cache: bool = True,
) -> dict:
    results = train_and_predict(draws, progress_cb, use_cache)
    from_cache = (CACHE_DIR / f"model_{_data_hash(draws)}.pkl").exists()

    base_rate = 5 / 39
    remaining = {n: results[n]["vote"] for n in ALL_NUMS if n not in killed}
    ranked = sorted(remaining.items(), key=lambda x: -x[1])

    detail = []
    for n, p_vote in ranked[:10]:
        r = results[n]
        detail.append({
            "號碼": f"{n:02d}",
            "綜合投票": f"{p_vote:.1%}",
            "XGBoost":  f"{r['xgb']:.1%}",
            "GradBoost":f"{r['gb']:.1%}",
            "RandomForest": f"{r['rf']:.1%}",
            "基準":     f"{base_rate:.1%}",
            "超出基準": f"+{p_vote - base_rate:.1%}" if p_vote > base_rate else f"{p_vote - base_rate:.1%}",
            "建議": "✅ 強烈" if p_vote > base_rate * 1.3 else (
                    "👍 推薦" if p_vote > base_rate * 1.1 else "⚠️ 普通"),
        })

    return {
        "top5":       sorted([n for n, _ in ranked[:5]]),
        "top6":       sorted([n for n, _ in ranked[:6]]),
        "top7":       sorted([n for n, _ in ranked[:7]]),
        "all_probs":  {n: results[n]["vote"] for n in ALL_NUMS},
        "ranked":     ranked,
        "detail":     detail,
        "from_cache": from_cache,
        "importance": results.get("__importance__", {}),
    }
