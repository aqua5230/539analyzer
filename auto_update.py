#!/usr/bin/env python3
"""
539 每日自動更新腳本
- 只抓最新1頁，合併進現有 CSV（不覆蓋歷史）
- 若有新資料則清除 ML/LSTM 快取
- 執行後寫入日誌 auto_update.log
"""

import csv
import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
LOG  = BASE / "auto_update.log"
CSV  = BASE / "539_history.csv"
CACHE = BASE / "model_cache"

os.chdir(BASE)


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_csv() -> tuple[list[dict], set[str]]:
    if not CSV.exists():
        return [], set()
    with open(CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return rows, {r["期數"] for r in rows}


def write_csv(rows: list[dict]):
    import re
    def sort_key(r):
        m = re.match(r"(\d{4})/(\d{2})/(\d{2})", r["期數"])
        return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (0, 0, 0)
    rows.sort(key=sort_key)
    with open(CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["期數","號碼1","號碼2","號碼3","號碼4","號碼5"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    log("=== 開始自動更新 ===")

    existing_rows, existing_dates = read_csv()
    log(f"現有資料：{len(existing_rows)} 筆")

    try:
        from scraper import fetch_latest
        new_records = fetch_latest()
        log(f"抓到最新一頁：{len(new_records)} 筆")
    except Exception as e:
        log(f"❌ 抓取失敗：{e}")
        sys.exit(1)

    # 只加入還沒有的期數
    added = [r for r in new_records if r["期數"] not in existing_dates]

    if added:
        log(f"✅ 新增 {len(added)} 筆：{[r['期數'] for r in added]}")
        all_rows = existing_rows + added
        write_csv(all_rows)
        log(f"已儲存，總計 {len(all_rows)} 筆")

        # 清除模型快取
        if CACHE.exists():
            cleared = 0
            for f in list(CACHE.glob("*.pkl")) + list(CACHE.glob("*.pt")):
                f.unlink(missing_ok=True)
                cleared += 1
            if cleared:
                log(f"🗑  清除 {cleared} 個模型快取（下次開 app 重新訓練）")
    else:
        log("ℹ️  今日暫無新資料")

    log("=== 更新結束 ===\n")


if __name__ == "__main__":
    main()
