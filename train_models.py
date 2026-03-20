"""
背景模型訓練腳本 (train_models.py)
用途：排程執行（如每晚開獎後），自動抓取最新資料並更新 ML/DL 模型快取。
"""
import os
import time
from pathlib import Path
from datetime import datetime

# 確保執行目錄在腳本所在位置
os.chdir(Path(__file__).parent)

from scraper import download_all
from analyzer import load_draws_from_csv
from analysis import recommend
from ml_predict import get_ml_recommendation, ML_AVAILABLE
from dl_predict import get_lstm_recommendation, DL_AVAILABLE


def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 開始執行背景更新作業...")

    # 1. 更新資料庫
    print("-> 📥 檢查並抓取最新歷史資料...")
    # 背景更新通常只需抓前 2 頁確保不漏接即可
    download_all("539_history.csv", max_pages=2)

    draws = load_draws_from_csv("539_history.csv")
    if len(draws) < 10:
        print("-> ❌ [錯誤] 資料不足，終止訓練。")
        return

    print(f"-> ✅ 成功載入 {len(draws)} 期資料。最新期數：{draws[-1].period}")

    # 2. 取得基礎統計推薦（為了拿到 killed 殺號名單）
    print("-> 🧮 計算基礎統計與殺號條件...")
    rec = recommend(draws)
    killed_set = set(rec.killed)

    # 3. 訓練 XGBoost 家族 (ML)
    if ML_AVAILABLE:
        print("-> 🤖 開始訓練 ML 模型 (XGBoost/RF/GB)...")
        start_time = time.time()
        # 強制 use_cache=False 以覆寫舊快取
        get_ml_recommendation(draws, killed_set, use_cache=False)
        print(f"-> ✅ ML 模型訓練完成，耗時 {time.time() - start_time:.1f} 秒。")
    else:
        print("-> ⚠️ [跳過] 偵測不到 ML 環境套件 (scikit-learn, xgboost)。")

    # 4. 訓練 LSTM 模型 (DL)
    if DL_AVAILABLE:
        print("-> 🧠 開始訓練 DL 模型 (LSTM+Attention)...")
        start_time = time.time()
        # 強制 use_cache=False 以覆寫舊快取
        get_lstm_recommendation(draws, killed_set, use_cache=False)
        print(f"-> ✅ DL 模型訓練完成，耗時 {time.time() - start_time:.1f} 秒。")
    else:
        print("-> ⚠️ [跳過] 偵測不到 PyTorch 環境套件。")

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🎉 所有背景訓練作業已完成！")


if __name__ == "__main__":
    main()
