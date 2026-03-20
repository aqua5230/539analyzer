"""
今彩539 歷史開獎資料爬蟲
資料來源：https://www.pilio.idv.tw/lto539/list.asp

抓取策略：
- 歷史頁面用 orderby=old（從舊到新，不受快取影響）
- 最新一頁直接用無參數 URL（避免 CDN 快取舊版本）
"""

from __future__ import annotations
import csv
import subprocess
import time
from pathlib import Path

from bs4 import BeautifulSoup

BASE_URL = "https://www.pilio.idv.tw/lto539/list.asp"


def _fetch_html(url: str) -> str:
    """用 curl 抓取頁面 HTML。"""
    result = subprocess.run(
        ["curl", "-s",
         "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
         url],
        capture_output=True, text=True, timeout=20
    )
    return result.stdout


def _parse_table(html: str) -> list[dict]:
    """從 HTML 解析 ltotable，回傳開獎紀錄列表。"""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="ltotable")
    if not table:
        return []

    records = []
    for row in table.find_all("tr"):
        date_cell = row.find("td", class_="date-cell")
        num_cell = row.find("td", class_="number-cell")
        if not date_cell or not num_cell:
            continue
        try:
            # 原始：第1行 "03/16"，第2行 "26(一)" → 組成 "2026/03/16(一)"
            parts_date = [t.strip().replace("\xa0", "") for t in date_cell.get_text("\n").split("\n") if t.strip()]
            if len(parts_date) >= 2:
                md = parts_date[0]   # "03/16"
                yw = parts_date[1]   # "26(一)"
                yy = yw[:2]          # "26"
                weekday = yw[2:]     # "(一)"
                date_str = f"20{yy}/{md}{weekday}"  # "2026/03/16(一)"
            else:
                date_str = parts_date[0] if parts_date else ""
            raw = num_cell.get_text(separator=",")
            parts = [p.strip().replace("\xa0", "").replace(" ", "") for p in raw.split(",")]
            nums = sorted([int(p) for p in parts if p.isdigit()][:5])
            if len(nums) < 5:
                continue
            records.append({
                "期數": date_str,
                "號碼1": nums[0], "號碼2": nums[1],
                "號碼3": nums[2], "號碼4": nums[3], "號碼5": nums[4],
            })
        except ValueError:
            continue
    return records


def get_total_pages() -> int:
    """取得總頁數（從無參數首頁解析「最末頁」連結）。"""
    html = _fetch_html(BASE_URL)
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if "indexpage=" in href and ("最末頁" in a.text or "last" in href.lower()):
            try:
                return int(href.split("indexpage=")[1].split("&")[0])
            except ValueError:
                pass
    return 253


def fetch_latest() -> list[dict]:
    """抓最新一頁（無參數 URL，永遠是最新資料）。"""
    return _parse_table(_fetch_html(BASE_URL))


def fetch_page_old(page: int) -> list[dict]:
    """用 orderby=old 抓指定頁（歷史資料，不受快取干擾）。"""
    url = f"{BASE_URL}?indexpage={page}&orderby=old"
    return _parse_table(_fetch_html(url))


def download_all(
    output_path: str = "539_history.csv",
    max_pages: int | None = None,
    delay: float = 0.4,
    progress_callback=None,
) -> int:
    """
    下載歷史資料存成 CSV（由舊到新）。

    max_pages=None → 全部歷史（約253頁）
    max_pages=N    → 只抓最近 N 頁
    """
    total_pages = get_total_pages()

    if max_pages:
        # 最新 N 頁：取 total_pages 末尾 N 頁（orderby=old）
        start = max(1, total_pages - max_pages + 1)
        pages_to_fetch = list(range(start, total_pages + 1))
    else:
        pages_to_fetch = list(range(1, total_pages + 1))

    # 先讀現有 CSV，避免覆蓋歷史資料
    all_records: list[dict] = []
    seen_dates: set[str] = set()
    _existing = Path(output_path)
    if _existing.exists():
        with open(_existing, newline="", encoding="utf-8-sig") as _f:
            for row in csv.DictReader(_f):
                if row["期數"] not in seen_dates:
                    seen_dates.add(row["期數"])
                    all_records.append(row)

    for idx, page in enumerate(pages_to_fetch, 1):
        records = fetch_page_old(page)
        for r in records:
            if r["期數"] not in seen_dates:
                seen_dates.add(r["期數"])
                all_records.append(r)
        if progress_callback:
            progress_callback(idx, len(pages_to_fetch))
        if idx < len(pages_to_fetch):
            time.sleep(delay)

    # 補最新一頁（無參數 URL，確保拿到最新資料）
    for r in fetch_latest():
        if r["期數"] not in seen_dates:
            seen_dates.add(r["期數"])
            all_records.append(r)

    # 依日期排序（格式 MM/DDYY(weekday) → 取 MM/DD 和 YY 排序）
    import re
    def sort_key(r):
        # 格式：2026/03/16(一)
        m = re.match(r"(\d{4})/(\d{2})/(\d{2})", r["期數"])
        if m:
            return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return (0, 0, 0)
    all_records.sort(key=sort_key)

    # 寫入 CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["期數", "號碼1", "號碼2", "號碼3", "號碼4", "號碼5"])
        writer.writeheader()
        writer.writerows(all_records)

    return len(all_records)


if __name__ == "__main__":
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else "5"

    if arg == "all":
        print("下載全部歷史資料（約253頁）...")
        def prog(cur, tot): print(f"\r  {cur}/{tot} 頁", end="", flush=True)
        n = download_all("539_history.csv", progress_callback=prog)
        print(f"\n✅ 共 {n} 筆 → 539_history.csv")
    else:
        pages = int(arg) if arg.isdigit() else 5
        print(f"下載最新 {pages} 頁...")
        def prog(cur, tot): print(f"\r  {cur}/{tot} 頁", end="", flush=True)
        n = download_all("539_history.csv", max_pages=pages, progress_callback=prog)
        print(f"\n✅ 共 {n} 筆 → 539_history.csv")
