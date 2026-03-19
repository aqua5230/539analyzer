"""
今彩539 Markov Chain 預測模組
原理：計算「上期出現號碼 → 下期各號碼」的條件轉移機率
方法：
  - 一階 Markov：P(Y | X 上期出現)
  - 多滯後加權：lag1×60% + lag2×30% + lag3×10%
"""

from __future__ import annotations
from collections import Counter, defaultdict
from analyzer import Draw

ALL_NUMS = list(range(1, 40))


def build_transition_matrix(draws: list[Draw]) -> dict[int, dict[int, float]]:
    """
    建立轉移矩陣：P(Y 下期出現 | X 本期出現)
    = 「X 和 Y 在相鄰兩期分別出現」的次數 / 「X 出現」的次數
    """
    appear_count: Counter = Counter()
    co_next: Counter = Counter()

    for i in range(len(draws) - 1):
        curr = set(draws[i].numbers)
        nxt  = set(draws[i + 1].numbers)
        for x in curr:
            appear_count[x] += 1
            for y in nxt:
                co_next[(x, y)] += 1

    matrix: dict[int, dict[int, float]] = {}
    for x in ALL_NUMS:
        total = appear_count.get(x, 1)
        matrix[x] = {y: co_next.get((x, y), 0) / total for y in ALL_NUMS}

    return matrix


def get_markov_scores(
    draws: list[Draw],
    matrix: dict[int, dict[int, float]] | None = None,
) -> dict[int, float]:
    """
    多滯後加權 Markov 評分：
    score[Y] = 0.60 × P(Y | lag1) + 0.30 × P(Y | lag2) + 0.10 × P(Y | lag3)
    """
    if matrix is None:
        matrix = build_transition_matrix(draws)

    lag_weights = [(1, 0.60), (2, 0.30), (3, 0.10)]
    scores: dict[int, float] = defaultdict(float)

    for lag, w in lag_weights:
        if len(draws) < lag + 1:
            continue
        lag_nums = draws[-lag].numbers
        for y in ALL_NUMS:
            prob = sum(matrix[x].get(y, 0) for x in lag_nums)
            scores[y] += prob * w

    return dict(scores)


def get_markov_recommendation(
    draws: list[Draw],
    killed: set[int],
) -> dict:
    """
    Markov Chain 推薦：回傳號碼機率、推薦組合、轉移分析
    """
    matrix   = build_transition_matrix(draws)
    scores   = get_markov_scores(draws, matrix)
    base_rate = 5 / 39

    # 排除殺號
    remaining = {n: scores[n] for n in ALL_NUMS if n not in killed}
    ranked    = sorted(remaining.items(), key=lambda x: -x[1])

    # 機率正規化（轉成類似機率的數值）
    max_score = max(scores.values()) if scores else 1.0
    all_probs = {n: round(scores[n] / max_score * base_rate * 2, 4) for n in ALL_NUMS}

    detail = []
    for n, s in ranked[:10]:
        detail.append({
            "號碼":      f"{n:02d}",
            "Markov分": f"{s:.4f}",
            "來源":      "、".join(
                f"{x:02d}→{n:02d}" for x in draws[-1].numbers
                if matrix[x].get(n, 0) > 0.15
            ) or "綜合",
        })

    # 上期各號碼最可能帶出的下期號碼
    last_nums  = draws[-1].numbers
    transition_top = {}
    for x in last_nums:
        top_next = sorted(
            [(y, matrix[x][y]) for y in ALL_NUMS if y != x],
            key=lambda t: -t[1]
        )[:5]
        transition_top[f"{x:02d}"] = [(f"{y:02d}", round(p, 3)) for y, p in top_next]

    return {
        "top5":           sorted([n for n, _ in ranked[:5]]),
        "top6":           sorted([n for n, _ in ranked[:6]]),
        "top7":           sorted([n for n, _ in ranked[:7]]),
        "all_probs":      all_probs,
        "scores":         scores,
        "ranked":         ranked,
        "detail":         detail,
        "transition_top": transition_top,
        "matrix":         matrix,
    }
