"""
今彩539 全方位分析模組 v5
整合：殺號 + 遺漏/頻率(指數衰減) + 奇偶 + 大小 + 和值 + 重號/連號
     + 位置分析 + 號碼配對 + 波段週期 + 尾數分析（NEW）
"""

from __future__ import annotations
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from analyzer import Draw, FORMULAS

# ── 評分參數常數 ──────────────────────────────
DECAY_HALF_LIFE  = 50    # 指數衰減半衰期（期數）
REPEAT_THRESHOLD = 0.5   # 重號機率門檻
CONSEC_THRESHOLD = 0.4   # 連號機率門檻
FREQ_WEIGHT      = 0.5   # 頻率權重
GAP_WEIGHT       = 0.4   # 遺漏期數權重
ODDEVEN_BONUS    = 2.0
BIGSMALL_BONUS   = 2.0
SUM_BONUS        = 3.0
REPEAT_BONUS     = 2.0
CONSEC_BONUS     = 1.0
POS_MAX_BONUS    = 3.0
PAIR_MAX_BONUS   = 3.0
CYCLE_MAX_BONUS  = 4.0
TAIL_BONUS       = 2.0   # 尾數加分
TAIL_TOP_N       = 3     # 取前N個高頻尾數
TAIL_LOOKBACK    = 30    # 尾數分析回溯期數


# ── 共用計算工具（避免重複掃描）──────────────

def _compute_position_freq(draws: list[Draw]) -> list[Counter]:
    pos_freq = [Counter() for _ in range(5)]
    for d in draws:
        for i, n in enumerate(d.numbers):
            pos_freq[i][n] += 1
    return pos_freq


def _compute_pair_counts(draws: list[Draw]) -> Counter:
    pair_cnt = Counter()
    for d in draws:
        for i, a in enumerate(d.numbers):
            for b in d.numbers[i+1:]:
                pair_cnt[(a, b)] += 1
    return pair_cnt


def _compute_decay_freq(draws: list[Draw]) -> dict[int, float]:
    """指數衰減頻率（越近期權重越高，半衰期 DECAY_HALF_LIFE 期）"""
    freq: dict[int, float] = defaultdict(float)
    n = len(draws)
    for i, d in enumerate(draws):
        w = math.exp((i - n + 1) / DECAY_HALF_LIFE)
        for num in d.numbers:
            freq[num] += w
    return dict(freq)


# ── 基礎統計 ──────────────────────────────────

def analyze_odd_even(draws):
    c = Counter(sum(1 for n in d.numbers if n % 2 == 1) for d in draws)
    t = len(draws)
    dist = {k: v/t for k, v in sorted(c.items())}
    best = max(dist, key=dist.get)
    return {"分布": dist, "最常見奇數個數": best, "機率": dist[best]}


def analyze_big_small(draws):
    c = Counter(sum(1 for n in d.numbers if n >= 20) for d in draws)
    t = len(draws)
    dist = {k: v/t for k, v in sorted(c.items())}
    best = max(dist, key=dist.get)
    return {"分布": dist, "最常見大號個數": best, "機率": dist[best]}


def analyze_sum_range(draws):
    s = sorted(sum(d.numbers) for d in draws)
    return {
        "平均和值": round(sum(s)/len(s), 1),
        "核心區間(25-75%)": (s[int(len(s)*.25)], s[int(len(s)*.75)]),
        "廣域區間(10-90%)": (s[int(len(s)*.10)], s[int(len(s)*.90)]),
    }


def analyze_repeat_numbers(draws):
    cnt = sum(1 for i in range(1, len(draws))
              if set(draws[i-1].numbers) & set(draws[i].numbers))
    t = len(draws) - 1
    avg = sum(len(set(draws[i-1].numbers) & set(draws[i].numbers))
              for i in range(1, len(draws))) / t
    return {"有重號機率": cnt/t, "平均重號個數": round(avg, 2)}


def analyze_consecutive(draws):
    cnt = sum(1 for d in draws
              if any(d.numbers[i+1] - d.numbers[i] == 1 for i in range(4)))
    return {"有連號機率": cnt/len(draws)}


def analyze_span(draws):
    spans = [d.numbers[-1] - d.numbers[0] for d in draws]
    return {"平均跨度": round(sum(spans)/len(spans), 1),
            "最常見跨度": Counter(spans).most_common(3)}


# ── 位置分析 ──────────────────────────────────

def analyze_positions(draws: list[Draw]) -> dict:
    pos_freq = _compute_position_freq(draws)
    result = {}
    for i in range(5):
        top5 = pos_freq[i].most_common(5)
        result[f"位置{i+1}"] = {
            "最常見": [n for n, _ in top5],
            "頻率": {n: round(cnt/len(draws), 3) for n, cnt in top5},
        }
    return result


def get_position_scores(draws: list[Draw]) -> dict[int, float]:
    pos_freq = _compute_position_freq(draws)
    scores: dict[int, float] = defaultdict(float)
    for i in range(5):
        for n, cnt in pos_freq[i].most_common(10):
            scores[n] += cnt / len(draws) * 2
    return dict(scores)


# ── 號碼配對分析 ──────────────────────────────

def analyze_pairs(draws: list[Draw], top_n: int = 20) -> list[tuple]:
    return _compute_pair_counts(draws).most_common(top_n)


def get_pair_scores(draws: list[Draw]) -> dict[int, float]:
    if len(draws) < 2:
        return {}
    latest_nums = set(draws[-1].numbers)
    pair_cnt = _compute_pair_counts(draws)
    scores: dict[int, float] = defaultdict(float)
    for (a, b), cnt in pair_cnt.items():
        rate = cnt / len(draws)
        if a in latest_nums:
            scores[b] += rate * 3
        if b in latest_nums:
            scores[a] += rate * 3
    return dict(scores)


# ── 波段週期分析 ──────────────────────────────

def analyze_cycle(draws: list[Draw]) -> dict[int, dict]:
    last_seen: dict[int, int] = {}
    intervals: dict[int, list] = defaultdict(list)
    for i, d in enumerate(draws):
        for n in d.numbers:
            if n in last_seen:
                intervals[n].append(i - last_seen[n])
            last_seen[n] = i

    total = len(draws)
    result = {}
    for n in range(1, 40):
        ivs = intervals.get(n, [])
        avg_iv = round(sum(ivs)/len(ivs), 1) if ivs else 8.0
        gap = total - 1 - last_seen.get(n, 0)
        overdue = gap >= avg_iv
        result[n] = {
            "平均間隔": avg_iv,
            "當前遺漏": gap,
            "已到期": overdue,
            "超期率": round((gap - avg_iv) / avg_iv, 2) if avg_iv > 0 else 0,
        }
    return result


def get_cycle_scores_from_info(cycle: dict[int, dict]) -> dict[int, float]:
    return {
        n: min(info["超期率"] * 2, CYCLE_MAX_BONUS) if info["已到期"] else 0.0
        for n, info in cycle.items()
    }


# ── 尾數分析（v5 新增）────────────────────────

def analyze_tail_digit(draws: list[Draw]) -> dict[int, float]:
    """統計各尾數（個位數 0~9）歷史出現頻率"""
    tail_cnt = Counter(n % 10 for d in draws for n in d.numbers)
    total = len(draws) * 5
    return {t: round(cnt/total, 4) for t, cnt in sorted(tail_cnt.items())}


def get_tail_scores(draws: list[Draw]) -> dict[int, float]:
    """近30期高頻尾數對應的號碼加分"""
    recent = draws[-TAIL_LOOKBACK:]
    tail_cnt = Counter(n % 10 for d in recent for n in d.numbers)
    top_tails = {t for t, _ in tail_cnt.most_common(TAIL_TOP_N)}
    return {n: TAIL_BONUS if n % 10 in top_tails else 0.0 for n in range(1, 40)}


# ── 綜合推薦 ─────────────────────────────────

@dataclass
class Recommendation:
    top5: list[int]
    top6: list[int]
    top7: list[int]
    killed: list[int]
    scores: dict[int, float]
    score_breakdown: dict[int, dict]
    stats: dict = field(default_factory=dict)


def recommend(draws: list[Draw]) -> Recommendation:
    """
    整合11種分析，給每個號碼算綜合分數：
    ①  殺號（6公式聯合排除）
    ②③ 遺漏+頻率（指數衰減，半衰期50期）
    ④  奇偶比適配
    ⑤  大小號適配
    ⑥  和值範圍適配
    ⑦  重號/連號加分
    ⑧  位置分析加分
    ⑨  號碼配對加分
    ⑩  波段週期加分
    ⑪  尾數分析加分（NEW v5）
    """
    if len(draws) < 10:
        raise ValueError("資料不足")

    latest = draws[-1]

    # ① 殺號
    killed: set[int] = set()
    for fn in FORMULAS.values():
        killed.update(fn(latest.numbers))
    remaining = [n for n in range(1, 40) if n not in killed]

    # 基礎統計
    oe = analyze_odd_even(draws)
    bs = analyze_big_small(draws)
    sr = analyze_sum_range(draws)
    rp = analyze_repeat_numbers(draws)
    cs = analyze_consecutive(draws)
    sp = analyze_span(draws)

    best_odd = oe["最常見奇數個數"]
    best_big = bs["最常見大號個數"]
    sum_lo, sum_hi = sr["核心區間(25-75%)"]
    has_repeat = rp["有重號機率"] > REPEAT_THRESHOLD
    has_consec  = cs["有連號機率"] > CONSEC_THRESHOLD

    # ②③ 指數衰減頻率 + 遺漏
    decay_freq = _compute_decay_freq(draws)
    last_seen: dict[int, int] = {}
    for i, d in enumerate(draws):
        for n in d.numbers:
            last_seen[n] = i
    total = len(draws)

    # ⑧⑨⑩⑪ 分析（cycle_info 只計算一次）
    pos_scores   = get_position_scores(draws)
    pair_scores  = get_pair_scores(draws)
    cycle_info   = analyze_cycle(draws)
    cycle_scores = get_cycle_scores_from_info(cycle_info)
    tail_scores  = get_tail_scores(draws)

    breakdown: dict[int, dict] = {}
    final_scores: dict[int, float] = {}

    for n in remaining:
        f    = decay_freq.get(n, 0.0)
        gap  = total - 1 - last_seen.get(n, 0)
        base = round(f * FREQ_WEIGHT + gap * GAP_WEIGHT, 2)

        is_odd = n % 2 == 1
        odd_b  = ODDEVEN_BONUS if (is_odd and best_odd >= 3) or (not is_odd and best_odd <= 2) else 0.0

        is_big = n >= 20
        big_b  = BIGSMALL_BONUS if (is_big and best_big >= 3) or (not is_big and best_big <= 2) else 0.0

        est_sum = sum(latest.numbers[:4]) + n
        sum_b   = SUM_BONUS if sum_lo <= est_sum <= sum_hi else 0.0

        rep_b  = REPEAT_BONUS if has_repeat and n in latest.numbers else 0.0
        con_b  = CONSEC_BONUS if has_consec and ((n-1) in latest.numbers or (n+1) in latest.numbers) else 0.0

        pos_b  = round(min(pos_scores.get(n, 0), POS_MAX_BONUS), 2)
        pair_b = round(min(pair_scores.get(n, 0), PAIR_MAX_BONUS), 2)
        cyc_b  = round(min(cycle_scores.get(n, 0), CYCLE_MAX_BONUS), 2)
        tail_b = round(tail_scores.get(n, 0), 2)

        total_s = round(base + odd_b + big_b + sum_b + rep_b + con_b
                        + pos_b + pair_b + cyc_b + tail_b, 2)

        final_scores[n] = total_s
        breakdown[n] = {
            "②③遺漏+頻率": base,
            "④奇偶":        odd_b,
            "⑤大小號":      big_b,
            "⑥和值":        sum_b,
            "⑦重號/連號":   rep_b + con_b,
            "⑧位置分析":    pos_b,
            "⑨號碼配對":    pair_b,
            "⑩波段週期":    cyc_b,
            "⑪尾數分析":    tail_b,
            "總分":          total_s,
        }

    ranked = sorted(final_scores.items(), key=lambda x: -x[1])

    return Recommendation(
        top5=sorted([n for n, _ in ranked[:5]]),
        top6=sorted([n for n, _ in ranked[:6]]),
        top7=sorted([n for n, _ in ranked[:7]]),
        killed=sorted(killed),
        scores=dict(ranked[:10]),
        score_breakdown=breakdown,
        stats={
            "奇偶": oe, "大小": bs, "和值": sr,
            "重號": rp, "連號": cs, "跨度": sp,
            "位置": analyze_positions(draws),
            "配對": analyze_pairs(draws),
            "週期": cycle_info,
            "尾數": analyze_tail_digit(draws),
        },
    )


# ── 步進回測 ─────────────────────────────────

def walk_forward_backtest(draws: list[Draw], n_test: int = 20) -> list[dict]:
    """
    步進回測：用前N-k期預測第N-k+1期，統計最近 n_test 期的實際命中率。
    """
    results = []
    for i in range(n_test, 0, -1):
        train = draws[:-i]
        actual = draws[-i]
        if len(train) < 20:
            continue
        try:
            r = recommend(train)
            results.append({
                "期數":   actual.period,
                "預測5碼": r.top5,
                "預測7碼": r.top7,
                "實際開獎": list(actual.numbers),
                "5碼命中": len(set(r.top5) & set(actual.numbers)),
                "7碼命中": len(set(r.top7) & set(actual.numbers)),
            })
        except Exception:
            pass
    return results
