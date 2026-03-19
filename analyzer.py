"""
台灣今彩539 殺號分析程式
方法來源：雙色球精準殺號定膽選號方法詳解（張委銘）
移植適配：今彩539（號碼範圍 01–39，每期開5碼）
"""

from __future__ import annotations
import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# ─────────────────────────────────────────────
# 資料結構
# ─────────────────────────────────────────────

@dataclass
class Draw:
    period: str
    numbers: list[int]   # 已排序，長度=5


# ─────────────────────────────────────────────
# 步驟1：資料輸入
# ─────────────────────────────────────────────

SAMPLE_DATA = """期數,號碼1,號碼2,號碼3,號碼4,號碼5
113001,03,12,19,27,35
113002,05,11,22,30,38
113003,02,09,18,25,33
113004,07,14,21,29,36
113005,01,08,17,24,32
113006,04,13,20,28,37
113007,06,10,23,31,39
113008,02,11,19,26,34
113009,03,08,16,23,31
113010,05,12,20,27,35
"""

def load_draws_from_csv(path: str) -> list[Draw]:
    """從 CSV 檔讀取歷史開獎紀錄。

    CSV 格式（標題行）：期數,號碼1,號碼2,號碼3,號碼4,號碼5
    """
    draws = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                nums = sorted([
                    int(row["號碼1"]), int(row["號碼2"]), int(row["號碼3"]),
                    int(row["號碼4"]), int(row["號碼5"])
                ])
                draws.append(Draw(period=row["期數"].strip(), numbers=nums))
            except (KeyError, ValueError):
                continue
    return draws


def load_draws_from_string(raw: str) -> list[Draw]:
    """從字串（CSV 格式）讀取，用於示範資料或使用者貼入。"""
    draws = []
    reader = csv.DictReader(io.StringIO(raw.strip()))
    for row in reader:
        try:
            nums = sorted([
                int(row["號碼1"]), int(row["號碼2"]), int(row["號碼3"]),
                int(row["號碼4"]), int(row["號碼5"])
            ])
            draws.append(Draw(period=row["期數"].strip(), numbers=nums))
        except (KeyError, ValueError):
            continue
    return draws


# ─────────────────────────────────────────────
# 步驟2 & 3：公式計算 + 轉換規則
# ─────────────────────────────────────────────

def unit_digit(n: int) -> int:
    """取絕對值後的個位數。"""
    return abs(n) % 10


def expand_unit(u: int) -> list[int]:
    """
    個位數 u → 生成所有今彩539範圍內尾數為 u 的號碼。
    範圍 01–39，個位 0 → [10, 20, 30]；其他 → [u, u+10, u+20, u+30] 去掉>39的
    """
    candidates = []
    for base in range(0, 40, 10):
        n = base + u
        if 1 <= n <= 39:
            candidates.append(n)
    return candidates


# ─────── 公式定義 ───────────────────────────

Formula = Callable[[list[int]], list[int]]


def formula_diff_5_1(nums: list[int]) -> list[int]:
    """公式A：第5碼 − 第1碼 → 個位數展開"""
    diff = nums[4] - nums[0]
    return expand_unit(unit_digit(diff))


def formula_sum_1_2(nums: list[int]) -> list[int]:
    """公式B：第1碼 + 第2碼 → 個位數展開"""
    s = nums[0] + nums[1]
    return expand_unit(unit_digit(s))


def formula_diff_4_2(nums: list[int]) -> list[int]:
    """公式C：第4碼 − 第2碼 → 個位數展開"""
    diff = nums[3] - nums[1]
    return expand_unit(unit_digit(diff))


def formula_sum_3_4(nums: list[int]) -> list[int]:
    """公式D：第3碼 + 第4碼 → 個位數展開"""
    s = nums[2] + nums[3]
    return expand_unit(unit_digit(s))


def formula_diff_5_3(nums: list[int]) -> list[int]:
    """公式E：第5碼 − 第3碼 → 個位數展開"""
    diff = nums[4] - nums[2]
    return expand_unit(unit_digit(diff))


def formula_sum_1_5(nums: list[int]) -> list[int]:
    """公式F：第1碼 + 第5碼 → 個位數展開"""
    s = nums[0] + nums[4]
    return expand_unit(unit_digit(s))


FORMULAS: dict[str, Formula] = {
    "A: 第5碼−第1碼": formula_diff_5_1,
    "B: 第1碼+第2碼": formula_sum_1_2,
    "C: 第4碼−第2碼": formula_diff_4_2,
    "D: 第3碼+第4碼": formula_sum_3_4,
    "E: 第5碼−第3碼": formula_diff_5_3,
    "F: 第1碼+第5碼": formula_sum_1_5,
}


# ─────────────────────────────────────────────
# 步驟4：歷史回測驗證
# ─────────────────────────────────────────────

@dataclass
class BacktestResult:
    formula_name: str
    total_rounds: int       # 可驗證的期數（需有下一期）
    success_count: int      # 殺號成功次數（下一期完全未開出）
    success_rate: float     # 成功率
    details: list[dict]     # 每期明細


def backtest(draws: list[Draw], formula_name: str, formula: Formula) -> BacktestResult:
    """
    殺號回測：
    - 用第 i 期開獎號碼計算殺號組
    - 若第 i+1 期的5個號碼與殺號組「完全無交集」→ 殺號成功
    """
    details = []
    success = 0

    for i in range(len(draws) - 1):
        current = draws[i]
        next_draw = draws[i + 1]

        killed = formula(current.numbers)
        overlap = set(killed) & set(next_draw.numbers)
        is_success = len(overlap) == 0

        if is_success:
            success += 1

        details.append({
            "期數": current.period,
            "下一期": next_draw.period,
            "本期號碼": current.numbers,
            "殺號組": sorted(killed),
            "下期號碼": next_draw.numbers,
            "命中數": len(overlap),
            "殺號成功": is_success,
        })

    total = len(draws) - 1
    rate = success / total if total > 0 else 0.0

    return BacktestResult(
        formula_name=formula_name,
        total_rounds=total,
        success_count=success,
        success_rate=rate,
        details=details,
    )


def run_all_backtests(draws: list[Draw]) -> list[BacktestResult]:
    results = []
    for name, fn in FORMULAS.items():
        results.append(backtest(draws, name, fn))
    return sorted(results, key=lambda r: r.success_rate, reverse=True)


# ─────────────────────────────────────────────
# 步驟5：印出報表（純 CLI 版）
# ─────────────────────────────────────────────

def print_report(results: list[BacktestResult]) -> None:
    print("\n" + "=" * 58)
    print("  台灣今彩539 殺號公式回測報表（成功率由高到低）")
    print("=" * 58)
    print(f"  {'公式名稱':<22} {'總期數':>6} {'成功次數':>8} {'成功率':>8}")
    print("-" * 58)
    for r in results:
        bar = "█" * int(r.success_rate * 20)
        print(
            f"  {r.formula_name:<22} {r.total_rounds:>6} "
            f"{r.success_count:>8} {r.success_rate:>7.1%}  {bar}"
        )
    print("=" * 58)
    print()
    print("  說明：殺號成功 = 下一期5個號碼完全不在殺號組中")
    print(f"  理論基準：若殺號組有4個號碼，理論殺號率 ≈ C(35,5)/C(39,5) ≈ 66.4%")
    print()


# ─────────────────────────────────────────────
# 主程式入口
# ─────────────────────────────────────────────

def main():
    import sys

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if not Path(csv_path).exists():
            print(f"[錯誤] 找不到檔案：{csv_path}")
            sys.exit(1)
        draws = load_draws_from_csv(csv_path)
        print(f"[資料] 從 {csv_path} 載入 {len(draws)} 期開獎紀錄")
    else:
        draws = load_draws_from_string(SAMPLE_DATA)
        print(f"[資料] 使用內建示範資料（{len(draws)} 期）")
        print("  提示：執行 python analyzer.py <your_data.csv> 可載入真實資料\n")

    if len(draws) < 2:
        print("[錯誤] 至少需要2期資料才能進行回測")
        sys.exit(1)

    results = run_all_backtests(draws)
    print_report(results)


if __name__ == "__main__":
    main()
