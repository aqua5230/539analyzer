"""台灣今彩539 全方位智慧選號 v5"""

import streamlit as st
import pandas as pd
import numpy as np
import io, os, json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from collections import Counter
from datetime import datetime
from math import comb

os.chdir(Path(__file__).parent)

from analyzer import load_draws_from_csv, load_draws_from_string, backtest, FORMULAS, SAMPLE_DATA
from analysis import recommend, analyze_cycle, walk_forward_backtest
from markov_predict import get_markov_recommendation

st.set_page_config(page_title="今彩539分析", page_icon="🎯", layout="wide")

st.markdown('<link href="https://fonts.googleapis.com/css2?family=LXGW+WenKai+TC:wght@300;400;700&family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">', unsafe_allow_html=True)


st.markdown("""
<style>
* { font-family: 'LXGW WenKai TC', 'Inter', -apple-system, sans-serif !important; }

/* === 等高欄位 === */
div[data-testid="stHorizontalBlock"] { align-items: stretch !important; }
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] { display:flex !important; flex-direction:column !important; }
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div[data-testid="stVerticalBlockBorderWrapper"] { flex:1 !important; display:flex !important; flex-direction:column !important; }
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div[data-testid="stVerticalBlockBorderWrapper"] > div { flex:1 !important; display:flex !important; flex-direction:column !important; }
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] { flex:1 !important; display:flex !important; flex-direction:column !important; }
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] > div { flex:1 !important; }

/* === Apple 磨砂玻璃卡片 === */
.rec-card {
    display:flex !important; flex-direction:column !important; justify-content:space-between !important;
    transition: all 0.5s cubic-bezier(0.25,0.1,0.25,1) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
}
.rec-card:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 20px 60px rgba(0,0,0,0.08) !important;
}
.trash-container {
    transition: all 0.5s cubic-bezier(0.25,0.1,0.25,1) !important;
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
}
.trash-container:hover { box-shadow: 0 12px 40px rgba(0,0,0,0.06) !important; transform: translateY(-2px); }
.toolbox-header {
    transition: all 0.5s cubic-bezier(0.25,0.1,0.25,1) !important;
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
}

/* === 球體動效 === */
@keyframes ball-bounce {
    0%,100% { transform: translateY(0); }
    50% { transform: translateY(-4px); }
}
.b-r:hover, .b-p:hover, .b-b:hover, .b-g:hover {
    animation: ball-bounce 0.5s cubic-bezier(0.25,0.1,0.25,1) !important;
    filter: brightness(1.08) !important;
}

/* === 裝飾動畫 === */
@keyframes float-slow {
    0%,100% { transform: translateY(0) rotate(0deg); }
    50% { transform: translateY(-15px) rotate(3deg); }
}
@keyframes pulse-soft {
    0%,100% { opacity: 0.4; transform: scale(1); }
    50% { opacity: 0.7; transform: scale(1.05); }
}

/* === Apple 磨砂共用 === */
.apple-glass {
    background: rgba(255,255,255,0.72) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    border: 1px solid rgba(255,255,255,0.6) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.04), inset 0 0 0 0.5px rgba(255,255,255,0.5) !important;
}

/* === Apple 風格背景 — 大色塊模糊 === */
.stApp {
    background: #eae6df !important;
    min-height: 100vh;
    position: relative;
}
.stApp::before {
    content: '';
    position: fixed;
    top: -20%; left: -10%;
    width: 50vw; height: 50vw;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,107,107,0.12) 0%, transparent 70%);
    filter: blur(60px);
    pointer-events: none;
    z-index: 0;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: -10%; right: -10%;
    width: 45vw; height: 45vw;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(78,205,196,0.1) 0%, transparent 70%);
    filter: blur(60px);
    pointer-events: none;
    z-index: 0;
}
[data-testid="stAppViewContainer"] { background: transparent !important; }
[data-testid="stAppViewContainer"] > section { background: transparent !important; }
[data-testid="stAppViewContainer"] > section > div { background: transparent !important; }
.block-container { background: transparent !important; padding-top: 0 !important; margin-top: -2rem !important; }

/* === 側邊欄（隱藏） === */
[data-testid="stSidebar"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
button[kind="header"] { display: none !important; }

/* === 隱藏 Streamlit UI === */
[data-testid="collapsedControl"] { display: none !important; }
#MainMenu { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
.stDeployButton { display: none !important; }

/* === 文字 === */
body, p, div, span, label { color: #333; font-size: 1.1rem; }
h1 { font-size: 2.4rem !important; }
h2 { font-size: 2rem !important; }
h3 { font-size: 1.6rem !important; }
h1,h2,h3,h4,h5 { color: #1d1d1f !important; font-weight: 900 !important; letter-spacing: -0.5px; }
h4 {
    font-size: 1.35rem !important; margin-bottom: 0.3rem !important;
    background: rgba(255,255,255,0.7) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border-left: 4px solid #4ECDC4 !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.2rem !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.03) !important;
}

/* === Subheader 樣式 === */
[data-testid="stSubheader"] {
    background: rgba(255,255,255,0.72);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid rgba(255,255,255,0.6);
    border-radius: 16px;
    padding: 1rem 1.4rem !important;
    margin: 1rem 0 0.5rem !important;
    font-size: 1.5rem !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.03);
    border-left: 4px solid #4ECDC4;
}

/* === Plotly 圖表容器 === */
[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,0.6);
    border: 1px solid rgba(0,0,0,0.04);
    border-radius: 16px;
    padding: 0.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.03);
}

/* === 分隔線淡化 === */
hr { border: none !important; height: 1px !important; background: linear-gradient(90deg, transparent, rgba(78,205,196,0.2), transparent) !important; margin: 1.5rem 0 !important; }

/* === Label 樣式 === */
[data-testid="stWidgetLabel"] p {
    color: #555 !important; font-weight: 600 !important; font-size: 1.05rem !important;
}

/* === 球體動畫 === */
@keyframes hl-orbit {
    0%   { top:10%; left:14%; }
    25%  { top:10%; left:52%; }
    50%  { top:50%; left:52%; }
    75%  { top:50%; left:14%; }
    100% { top:10%; left:14%; }
}
@keyframes glow-pulse {
    0%,100% { box-shadow:0 2px 8px rgba(0,0,0,0.08); }
    50% { box-shadow:0 4px 16px rgba(0,0,0,0.12); }
}

/* === 球體樣式 === */
.b-r,.b-p,.b-b,.b-g {
    border-radius: 50% !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-weight: 900 !important;
    position: relative !important;
    overflow: hidden !important;
    color: #fff !important;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
    cursor: default;
}
.b-r::before,.b-p::before,.b-b::before,.b-g::before {
    content: '';
    position: absolute;
    width: 42%; height: 42%;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.6) 0%, rgba(255,255,255,0.15) 60%, transparent 80%);
    filter: blur(2px);
    animation: hl-orbit 3s ease-in-out infinite;
    pointer-events: none;
}
.b-r {
    background: radial-gradient(circle at 35% 30%, #FF8A80 0%, #FF5252 40%, #D32F2F 100%);
    box-shadow: 0 3px 10px rgba(211,47,47,0.3), inset 0 -2px 4px rgba(0,0,0,0.1);
}
.b-p {
    background: radial-gradient(circle at 35% 30%, #CE93D8 0%, #AB47BC 40%, #7B1FA2 100%);
    box-shadow: 0 3px 10px rgba(123,31,162,0.3), inset 0 -2px 4px rgba(0,0,0,0.1);
}
.b-b {
    background: radial-gradient(circle at 35% 30%, #80CBC4 0%, #26A69A 40%, #00796B 100%);
    box-shadow: 0 3px 10px rgba(0,121,107,0.3), inset 0 -2px 4px rgba(0,0,0,0.1);
}
.b-g {
    background: radial-gradient(circle at 35% 30%, #90CAF9 0%, #42A5F5 40%, #1565C0 100%);
    box-shadow: 0 3px 10px rgba(21,101,192,0.3), inset 0 -2px 4px rgba(0,0,0,0.1);
}

/* === Tab === */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px; background: #f5f5f5; border-radius: 12px; padding: 4px;
    border: 1px solid #eee;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 10px; padding: 10px 20px; font-size: 1rem;
    font-weight: 600; color: #888; background: transparent; border: none;
}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg,#4ECDC4,#26A69A); color: #fff !important;
    box-shadow: 0 4px 12px rgba(78,205,196,0.3);
    border: none;
}

/* === Metric === */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.72); border: 1px solid rgba(255,255,255,0.6);
    border-left: 4px solid #4ECDC4;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.03);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    transition: all 0.5s cubic-bezier(0.25,0.1,0.25,1);
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] { color: #888 !important; font-size: 0.9rem !important; letter-spacing: 1px; font-weight: 700 !important; }
[data-testid="stMetricValue"] { color: #1d1d1f !important; font-size: 2rem !important; font-weight: 900 !important; letter-spacing: -1px; }

/* === Expander === */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.72); border: 1px solid rgba(255,255,255,0.6);
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.03);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
}
details > summary::marker { display: none !important; }
details > summary::-webkit-details-marker { display: none !important; }
[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
    display: none !important; visibility: hidden !important;
    width: 0 !important; height: 0 !important; overflow: hidden !important; font-size: 0 !important;
}
[data-testid="stExpander"] summary p {
    font-size: 1.05rem !important; color: #555 !important; font-weight: 600 !important;
}

/* === DataFrame === */
[data-testid="stDataFrame"] {
    border: 1px solid #eee; border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* === Input === */
[data-testid="stTextInput"] input {
    background: #f8f9fa !important; border: 1px solid #e0e0e0 !important;
    border-radius: 10px !important; color: #333 !important; font-size: 0.9rem !important;
}

/* === Button === */
[data-testid="stButton"] button {
    background: linear-gradient(135deg,#4ECDC4,#44B8AC); border: none;
    border-radius: 10px;
    color: #fff; font-weight: 700; font-size: 0.85rem;
    box-shadow: 0 3px 12px rgba(78,205,196,0.3);
    transition: all 0.3s ease;
}
[data-testid="stButton"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(78,205,196,0.4);
}

/* === Select Box === */
[data-baseweb="select"] > div {
    background: #f8f9fa !important; border: 1px solid #e0e0e0 !important;
    border-radius: 10px !important; color: #333 !important;
}
[data-baseweb="select"] [data-baseweb="tag"] { background: #4ECDC4 !important; }
[data-baseweb="popover"] [role="listbox"] {
    background: #fff !important; border: 1px solid #eee !important;
    border-radius: 10px !important; box-shadow: 0 8px 30px rgba(0,0,0,0.08) !important;
}
[data-baseweb="popover"] [role="option"] { color: #333 !important; }
[data-baseweb="popover"] [role="option"][aria-selected="true"],
[data-baseweb="popover"] [role="option"]:hover {
    background: rgba(78,205,196,0.12) !important;
}

/* === Slider === */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #4ECDC4 !important; border-color: #4ECDC4 !important;
}
[data-testid="stSlider"] div[data-testid="stThumbValue"] { color: #4ECDC4 !important; }
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {
    background: #4ECDC4 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:last-child > div {
    background: #e0e0e0 !important;
}

/* hr 已在上方定義 */
[data-testid="stCaptionContainer"] { color: #999 !important; font-size: 0.9rem; }
[data-testid="stAlert"] { border-radius: 10px; border: 1px solid #eee; }
[data-testid="stRadio"] label { color: #555 !important; font-size: 1rem; }

.stMarkdown + .stMarkdown { margin-top: 0.3rem; }
div[data-testid="stVerticalBlock"] > div { margin-bottom: 0.2rem; }

/* === 平板 === */
@media (max-width: 1024px) and (min-width: 769px) {
    .block-container { padding-left: 1.2rem !important; padding-right: 1.2rem !important; }
    .b-r, .b-p, .b-b, .b-g { width: 2.4rem !important; height: 2.4rem !important; font-size: 0.9rem !important; }
    button[data-baseweb="tab"] { padding: 6px 14px !important; font-size: 0.78rem !important; }
}

/* === 手機 === */
@media (max-width: 768px) {
    .block-container { padding-left: 0.6rem !important; padding-right: 0.6rem !important; padding-top: 0 !important; }
    body, p, label { font-size: 1.05rem !important; }
    h4 { font-size: 1.2rem !important; }
    div[style*="font-size:7rem"] { font-size: 4rem !important; letter-spacing: 8px !important; }

    /* 推薦卡片 — 橫排一行，球+信心並排 */
    .rec-card {
        padding: 1rem 1.2rem !important; border-radius: 14px !important;
        margin-bottom: 0.5rem !important; min-height: auto !important; height: auto !important;
        flex-direction: row !important; align-items: center !important;
        gap: 1rem !important;
    }
    .rec-card > div { display: flex !important; flex-direction: row !important; align-items: center !important; gap: 1rem !important; flex-wrap: wrap !important; }
    .b-r, .b-p, .b-b, .b-g { width: 2.8rem !important; height: 2.8rem !important; font-size: 1rem !important; }

    div[data-baseweb="tab-list"] { gap: 2px !important; }
    button[data-baseweb="tab"] { padding: 8px 12px !important; font-size: 0.85rem !important; }
    div[style*="padding:1.8rem"] { padding: 1rem 0.8rem 0.8rem !important; }
    div[style*="min-width:120px"] { min-width: 80px !important; max-width: none !important; flex: 1 1 calc(33% - 8px) !important; padding: 10px 8px !important; }
    [data-testid="stDataFrame"] { overflow-x: auto !important; }
    [data-testid="stExpander"] { margin-top: 0.3rem !important; margin-bottom: 0.3rem !important; }
    [data-testid="stExpander"] summary p { font-size: 1rem !important; }
    div[data-testid="stHorizontalBlock"] { gap: 0.4rem !important; }
    div[style*="grid-template-columns"] { grid-template-columns: repeat(2, 1fr) !important; }
    div[style*="grid-template-columns"] > div:last-child:nth-child(odd) { grid-column: 1 / -1; }
    .trash-container, .toolbox-header { padding: 1rem !important; }
    .trash-ball { width: 2.4rem !important; height: 2.4rem !important; margin: 4px !important; }
    .trash-num { font-size: 0.9rem !important; }
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] { margin-bottom: 0.3rem !important; }
    .latest-ball { width: 3rem !important; height: 3rem !important; font-size: 1.15rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.75rem !important; }
    [data-testid="stCaptionContainer"] { font-size: 0.85rem !important; }
}

/* === 小螢幕 === */
@media (max-width: 480px) {
    .block-container { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
    div[style*="min-width:120px"] { flex: 1 1 calc(30% - 6px) !important; padding: 8px 6px !important; }
}

/* === 最新開獎球 RWD === */
.latest-ball {
    display: inline-flex !important; align-items: center; justify-content: center;
    width: 3.6rem; height: 3.6rem; border-radius: 50%; font-weight: 900; font-size: 1.35rem;
}
@media (max-width: 768px) {
    .latest-ball { width: 2.8rem !important; height: 2.8rem !important; font-size: 1.05rem !important; }
}
@media (max-width: 480px) {
    .latest-ball { width: 2.5rem !important; height: 2.5rem !important; font-size: 0.95rem !important; }
}

</style>
""", unsafe_allow_html=True)

_COLOR_CLASS = {
    "#c0392b": "b-r", "#8b6543": "b-r", "#e67e22": "b-r", "#f39c12": "b-r",
    "#8e44ad": "b-p", "#6b6358": "b-p", "#6c3483": "b-p", "#5b2c6f": "b-p",
    "#7d3c98": "b-p", "#9b59b6": "b-p", "#d35400": "b-r", "#a04000": "b-r",
    "#1a5276": "b-b", "#2980b9": "b-b", "#2471a3": "b-b", "#1abc9c": "b-b",
    "#16a085": "b-b", "#0e6655": "b-b", "#27ae60": "b-b",
}

def num_ball(nums, color, size="2.5rem"):
    cls = _COLOR_CLASS.get(color, "b-g")
    fs = "1.05rem" if size == "2.5rem" else "0.9rem"
    return "　".join(
        f"<span class='{cls}' style='width:{size};height:{size};font-size:{fs};'>{n:02d}</span>"
        for n in nums)


TOTAL_COMBOS = comb(39, 5)  # 575,757


def calc_hit_prob(n_select: int) -> tuple[int, float]:
    """回傳 (需買幾注, 全中機率%)"""
    tickets = comb(n_select, 5)
    pct = tickets / TOTAL_COMBOS * 100
    return tickets, pct


def _dark_layout(title="", height=320, **kwargs):
    """統一的 Plotly 佈局"""
    layout = dict(
        title=dict(text=title, font=dict(size=15, color="#1d1d1f", family="LXGW WenKai TC")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#555", family="LXGW WenKai TC, Inter, sans-serif"),
        height=height,
        margin=dict(t=50, b=20, l=40, r=20),
        xaxis=dict(gridcolor="rgba(0,0,0,0.05)", zerolinecolor="rgba(0,0,0,0.08)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.05)", zerolinecolor="rgba(0,0,0,0.08)"),
    )
    layout.update(kwargs)
    return layout


# ══════════════════════════════════════════
# 側邊欄
# ══════════════════════════════════════════
# ── 自動載入資料（無側邊欄）──
draws = []
cached = Path("539_history.csv")
if cached.exists():
    draws = load_draws_from_csv(str(cached))

if len(draws) < 10:
    # 沒有快取時自動抓取
    from scraper import download_all
    with st.spinner("首次載入，正在抓取歷史資料..."):
        download_all("539_history.csv", max_pages=20)
    draws = load_draws_from_csv("539_history.csv")
    if len(draws) < 10:
        st.error("資料不足，請稍後重試。")
        st.stop()

# ── 快取 recommend()（資料不變就不重算）──────
draws_key = f"{len(draws)}_{draws[-1].period}"
if st.session_state.get("_rec_key") != draws_key:
    with st.spinner("分析中..."):
        rec = recommend(draws)
    st.session_state["_rec"] = rec
    st.session_state["_rec_key"] = draws_key
    st.session_state.pop("ml_result", None)
else:
    rec = st.session_state["_rec"]

latest = draws[-1]

# ══════════════════════════════════════════
# 頂部總結卡片
# ══════════════════════════════════════════
# 若有 ML 結果，顯示三方共同推薦
ml_badge = ""
if "quad_top5" in st.session_state:
    _q5 = st.session_state["quad_top5"]
    ml_badge = f"<div style='color:#f39c12;font-size:0.85rem;margin-top:0.5rem'>🏆 四方集成推薦：{'  '.join(f'<b>{n:02d}</b>' for n in _q5)}</div>"
elif "lstm_result" in st.session_state and "ml_result" in st.session_state:
    _ml = st.session_state["ml_result"]
    _lstm = st.session_state["lstm_result"]
    _triple_base = 5/39
    _all_stat = {n: rec.score_breakdown[n]["總分"] for n in rec.score_breakdown}
    _max_s = max(_all_stat.values()) or 1
    _tc = {}
    for _n in _all_stat:
        _s = _all_stat[_n] / _max_s
        _m = min(_ml["all_probs"].get(_n, 0) / (_triple_base * 2), 1.0)
        _l = min(_lstm["all_probs"].get(_n, 0) / (_triple_base * 2), 1.0)
        _tc[_n] = _s * 0.33 + _m * 0.33 + _l * 0.34
    _tc5 = sorted([n for n, _ in sorted(_tc.items(), key=lambda x: -x[1])[:5]])
    ml_badge = f"<div style='color:#f39c12;font-size:0.85rem;margin-top:0.5rem'>🏆 三方集成推薦：{'  '.join(f'<b>{n:02d}</b>' for n in _tc5)}</div>"
elif "ml_result" in st.session_state:
    _ml_c = st.session_state["ml_result"]
    _ov = sorted(set(rec.top5) & set(_ml_c["top5"]))
    if _ov:
        ml_badge = f"<div style='color:#f39c12;font-size:0.8rem;margin-top:0.5rem'>🏆 統計+ML雙重認可：{'  '.join(f'{n:02d}' for n in _ov)}</div>"

_t5, _p5 = calc_hit_prob(5)
_t7, _p7 = calc_hit_prob(7)
_t10, _p10 = calc_hit_prob(10)

# ── 信心分數計算
_sb = rec.score_breakdown
_max_s = max((v["總分"] for v in _sb.values()), default=1) or 1
def _conf(nums): return round(sum(_sb.get(n,{}).get("總分",0) for n in nums) / (len(nums) * _max_s) * 100)
_conf5, _conf6, _conf7 = _conf(rec.top5), _conf(rec.top6), _conf(rec.top7)

# ── 置中標題（帶裝飾）
st.markdown('''
<div style="text-align:center;padding:0 0 0.5rem;position:relative">
  <!-- 裝飾圓球 -->
  <div style='position:absolute;top:-5px;left:8%;width:40px;height:40px;border-radius:50%;background:rgba(255,107,107,0.12);animation:float-slow 4s ease-in-out infinite'></div>
  <div style='position:absolute;top:10px;right:10%;width:25px;height:25px;border-radius:50%;background:rgba(78,205,196,0.15);animation:float-slow 5s ease-in-out infinite 1s'></div>
  <div style='position:absolute;bottom:0;left:20%;width:18px;height:18px;border-radius:50%;background:rgba(132,94,194,0.1);animation:float-slow 3.5s ease-in-out infinite 0.5s'></div>
  <div style='position:absolute;top:5px;right:25%;width:12px;height:12px;border-radius:50%;background:rgba(255,199,95,0.2);animation:pulse-soft 3s ease-in-out infinite'></div>
  <!-- 主標題 -->
  <div style='font-size:7rem;font-weight:900;letter-spacing:14px;text-align:center;line-height:1.1;position:relative'>
    <span style="color:#FF6B6B">今</span><span style="color:#4ECDC4">彩</span>
    <span style="color:#845EC2;margin-left:4px">5</span><span style="color:#FF9671">3</span><span style="color:#FFC75F">9</span>
  </div>
  <div style='width:60px;height:4px;background:linear-gradient(90deg,#FF6B6B,#4ECDC4,#845EC2,#FFC75F);margin:10px auto 8px;border-radius:3px'></div>
  <div style='font-size:1.1rem;font-weight:600;color:#aaa;letter-spacing:6px;text-align:center'>數據分析 · 智慧選號</div>
</div>
''', unsafe_allow_html=True)

# ── 頂部：期數 + 上期號碼
_ball_colors = ["#FF6B6B","#FF9671","#FFC75F","#4ECDC4","#845EC2"]
_latest_balls = ""
for _i, _n in enumerate(latest.numbers):
    _bc = _ball_colors[_i % len(_ball_colors)]
    _latest_balls += "<span class='latest-ball' style='display:inline-flex;align-items:center;justify-content:center;border-radius:50%;background:" + _bc + ";color:#fff;font-size:1.35rem;font-weight:900;box-shadow:0 4px 16px " + _bc + "40;border:2px solid rgba(255,255,255,0.8);text-shadow:0 1px 2px rgba(0,0,0,0.2)'>" + f"{_n:02d}" + "</span>"

st.markdown(f"""
<div style='background:rgba(255,255,255,0.72);backdrop-filter:blur(20px) saturate(180%);-webkit-backdrop-filter:blur(20px) saturate(180%);border:1px solid rgba(255,255,255,0.6);border-radius:24px;padding:2rem 1.5rem 1.8rem;margin-bottom:10px;
            box-shadow:0 8px 32px rgba(0,0,0,0.04),inset 0 0 0 0.5px rgba(255,255,255,0.5);position:relative;overflow:hidden'>
  <div style='position:absolute;top:-20px;right:-20px;width:100px;height:100px;border-radius:50%;background:rgba(255,199,95,0.08)'></div>
  <div style='position:absolute;bottom:-15px;left:-15px;width:80px;height:80px;border-radius:50%;background:rgba(78,205,196,0.06)'></div>
  <div style='text-align:center;position:relative'>
    <div style='display:inline-block;background:rgba(78,205,196,0.1);color:#4ECDC4;font-size:0.7rem;font-weight:700;letter-spacing:2px;padding:3px 14px;border-radius:99px;margin-bottom:10px'>已分析 {len(draws)} 期</div>
    <div style='color:#1d1d1f;font-size:1.3rem;font-weight:900;letter-spacing:4px;margin-bottom:4px'>最新開獎號碼</div>
    <div style='color:#bbb;font-size:0.85rem;font-weight:600;margin-bottom:16px'>{latest.period}</div>
    <div style='display:flex;gap:12px;justify-content:center;flex-wrap:wrap'>
      {_latest_balls}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── 三欄球號
# ── 計算最高信心分數欄
_best_conf = max(_conf5, _conf6, _conf7)

_ring_counter = [0]
def _conf_ring_html(conf, is_best=False):
    _ring_counter[0] += 1
    if conf >= 70:
        _desc = "推薦"
        _color1 = "#4ECDC4"
    elif conf >= 50:
        _desc = "⚡ 值得一試"
        _color1 = "#FF9671"
    else:
        _desc = "❄️ 謹慎參考"
        _color1 = "#C4C4C4"
    _ring_size = "90px" if is_best else "70px"
    _font_size = "1.4rem" if is_best else "1rem"
    return (
        "<div style='display:flex;align-items:center;gap:12px;margin-top:12px'>"
        "  <div style='position:relative;width:" + _ring_size + ";height:" + _ring_size + ";flex-shrink:0'>"
        "    <div style='width:100%;height:100%;border-radius:50%;"
        "      background:conic-gradient(" + _color1 + " 0deg," + _color1 + " " + str(round(conf * 3.6)) + "deg,#f0f0f0 " + str(round(conf * 3.6)) + "deg,#f0f0f0 360deg);"
        "      -webkit-mask:radial-gradient(farthest-side,transparent calc(100% - 10px),#000 calc(100% - 9px));mask:radial-gradient(farthest-side,transparent calc(100% - 10px),#000 calc(100% - 9px))'></div>"
        "    <div style='position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);"
        "                color:#1d1d1f;font-size:" + _font_size + ";font-weight:900;text-align:center;line-height:1'>"
        "      " + str(conf) + "<span style='font-size:0.55rem;color:#999'>%</span>"
        "    </div>"
        "  </div>"
        "  <div>"
        "    <div style='color:#555;font-size:0.72rem;font-weight:600;letter-spacing:1px'>信心指數</div>"
        "    <div style='color:#999;font-size:0.62rem;margin-top:2px'>" + _desc + "</div>"
        "  </div>"
        "</div>"
    )

_c5, _c6, _c7 = st.columns(3)
with _c5:
    _is_best5 = (_best_conf == _conf5)
    if _is_best5:
        _border5 = "border:2px solid #FF6B6B;box-shadow:0 8px 25px rgba(255,107,107,0.12)"
        _crown5 = "<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px'><span style='font-size:1.1rem'>🏆</span>"
        _badge5 = "</div><div style='text-align:left;margin-top:10px'><span style='background:linear-gradient(135deg,#FF6B6B,#FF9671);color:#fff;font-size:0.65rem;font-weight:900;padding:4px 14px;border-radius:99px;letter-spacing:1px'>首選推薦</span></div>"
        _bg5 = "background:linear-gradient(135deg,#fff 0%,#fff5f5 100%)"
    else:
        _border5 = "border:1px solid rgba(0,0,0,0.06)"
        _crown5 = "<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px'><span style='font-size:1.1rem;opacity:0'>🏆</span>"
        _badge5 = "</div>"
        _bg5 = "background:#fff"
    _ball5_size = "3.2rem" if _is_best5 else "2.8rem"
    _ball5_font = "1.1rem" if _is_best5 else "1rem"
    _balls5_html = "".join("<span class='b-r' style='width:" + _ball5_size + ";height:" + _ball5_size + ";font-size:" + _ball5_font + ";'>" + f"{n:02d}" + "</span>" for n in rec.top5)
    _ring5_html = _conf_ring_html(_conf5, _is_best5)
    _html5 = ("<div class='rec-card' style='" + _bg5 + ";" + _border5 + ";border-radius:20px;padding:1.4rem;margin-top:8px;position:relative;overflow:hidden'>"
        + "<div style='position:relative;z-index:1'>"
        + _crown5
        + "<span style='color:#FF6B6B;font-size:0.75rem;font-weight:700;letter-spacing:1px'>五號中獎王（Pick 5）</span></div>"
        + "<div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:12px'>" + _balls5_html + "</div>"
        + _ring5_html + _badge5
        + "</div>")
    st.markdown(_html5, unsafe_allow_html=True)
with _c6:
    _is_best6 = (_best_conf == _conf6 and not (_best_conf == _conf5))
    if _is_best6:
        _border6 = "border:2px solid #845EC2;box-shadow:0 8px 25px rgba(132,94,194,0.12)"
        _crown6 = "<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px'><span style='font-size:1.1rem'>🏆</span>"
        _badge6 = "</div><div style='text-align:left;margin-top:10px'><span style='background:linear-gradient(135deg,#845EC2,#B39CD0);color:#fff;font-size:0.65rem;font-weight:900;padding:4px 14px;border-radius:99px;letter-spacing:1px'>首選推薦</span></div>"
        _bg6 = "background:linear-gradient(135deg,#fff 0%,#f8f5ff 100%)"
    else:
        _border6 = "border:1px solid rgba(0,0,0,0.06)"
        _crown6 = "<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px'><span style='font-size:1.1rem;opacity:0'>🏆</span>"
        _badge6 = "</div>"
        _bg6 = "background:#fff"
    _ball6_size = "3.2rem" if _is_best6 else "2.6rem"
    _ball6_font = "1.1rem" if _is_best6 else "0.95rem"
    _balls6_html = "".join("<span class='b-p' style='width:" + _ball6_size + ";height:" + _ball6_size + ";font-size:" + _ball6_font + ";'>" + f"{n:02d}" + "</span>" for n in rec.top6)
    _ring6_html = _conf_ring_html(_conf6, _is_best6)
    _html6 = ("<div class='rec-card' style='" + _bg6 + ";" + _border6 + ";border-radius:20px;padding:1.4rem;margin-top:8px;position:relative;overflow:hidden'>"
        + "<div style='position:relative;z-index:1'>"
        + _crown6
        + "<span style='color:#845EC2;font-size:0.75rem;font-weight:700;letter-spacing:1px'>六六大順組（Pick 6）</span></div>"
        + "<div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:12px'>" + _balls6_html + "</div>"
        + _ring6_html + _badge6
        + "</div>")
    st.markdown(_html6, unsafe_allow_html=True)
with _c7:
    _is_best7 = (_best_conf == _conf7 and not (_best_conf == _conf5) and not (_best_conf == _conf6))
    if _is_best7:
        _border7 = "border:2px solid #4ECDC4;box-shadow:0 8px 25px rgba(78,205,196,0.12)"
        _crown7 = "<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px'><span style='font-size:1.1rem'>🏆</span>"
        _badge7 = "</div><div style='text-align:left;margin-top:10px'><span style='background:linear-gradient(135deg,#4ECDC4,#26A69A);color:#fff;font-size:0.65rem;font-weight:900;padding:4px 14px;border-radius:99px;letter-spacing:1px'>首選推薦</span></div>"
        _bg7 = "background:linear-gradient(135deg,#fff 0%,#f0faf9 100%)"
    else:
        _border7 = "border:1px solid rgba(0,0,0,0.06)"
        _crown7 = "<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px'><span style='font-size:1.1rem;opacity:0'>🏆</span>"
        _badge7 = "</div>"
        _bg7 = "background:#fff"
    _ball7_size = "3.2rem" if _is_best7 else "2.6rem"
    _ball7_font = "1.1rem" if _is_best7 else "0.95rem"
    _balls7_html = "".join("<span class='b-b' style='width:" + _ball7_size + ";height:" + _ball7_size + ";font-size:" + _ball7_font + ";'>" + f"{n:02d}" + "</span>" for n in rec.top7)
    _ring7_html = _conf_ring_html(_conf7, _is_best7)
    _html7 = ("<div class='rec-card' style='" + _bg7 + ";" + _border7 + ";border-radius:20px;padding:1.4rem;margin-top:8px;position:relative;overflow:hidden'>"
        + "<div style='position:relative;z-index:1'>"
        + _crown7
        + "<span style='color:#4ECDC4;font-size:0.75rem;font-weight:700;letter-spacing:1px'>七星連珠組（Pick 7）</span></div>"
        + "<div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:12px'>" + _balls7_html + "</div>"
        + _ring7_html + _badge7
        + "</div>")
    st.markdown(_html7, unsafe_allow_html=True)

# 等高 JS（透過 iframe 注入到 parent）
import streamlit.components.v1 as components
components.html("""
<script>
(function(){
  function eq(){
    var doc=window.parent.document;
    // 手機版（寬度<768）不做等高，避免大片空白
    if(doc.documentElement.clientWidth<768)return;
    var cards=doc.querySelectorAll('.rec-card');
    if(cards.length<2)return;
    var mx=0;
    cards.forEach(function(c){c.style.minHeight='auto';});
    cards.forEach(function(c){var h=c.offsetHeight;if(h>mx)mx=h;});
    cards.forEach(function(c){c.style.minHeight=mx+'px';});
  }
  setTimeout(eq,800);
  setTimeout(eq,2000);
})();
</script>
""", height=0)

# ── 排除號碼（互動式垃圾桶）
killed_list = sorted(rec.killed)
killed_reasons = {}
for n in killed_list:
    reasons = []
    freq_total = sum(1 for d in draws for num in d.numbers if num == n)
    last_seen = next((i for i, d in enumerate(reversed(draws)) if n in d.numbers), None)
    avg_gap = len(draws) / max(freq_total, 1)
    if last_seen is not None and last_seen > 25:
        reasons.append(f"🧊 冷門號碼，最近 {last_seen} 期沒開過")
    if freq_total < len(draws) * 0.08:
        reasons.append(f"📉 歷史出現率極低（{freq_total}/{len(draws)} 期）")
    if avg_gap > 10:
        reasons.append(f"⏳ 平均每 {avg_gap:.0f} 期才開一次")
    if not reasons:
        reasons.append("🔢 六大殺號公式判定本期不利")
    killed_reasons[n] = reasons

# 垃圾桶號碼球 HTML
_trash_balls = ""
for n in killed_list:
    _reason_html = "<br>".join(killed_reasons[n])
    _trash_balls += f"<div class='trash-ball'><span class='trash-num'>{n:02d}</span><div class='trash-tooltip'><div style='font-weight:700;margin-bottom:6px;color:#FF6B6B'>號碼 {n:02d} 被排除</div><div style='font-size:0.75rem;line-height:1.6'>{_reason_html}</div></div></div>"

st.markdown(f"""
<style>
.trash-container {{
  background: #fff;
  border: 1px solid #eee;
  border-radius: 20px;
  padding: 1.2rem 1.4rem;
  margin-bottom: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}}
.trash-header {{
  display: flex; align-items: center; gap: 0.6rem;
  margin-bottom: 0.8rem; cursor: default;
}}
.trash-icon {{
  font-size: 1.4rem;
  animation: trash-shake 2s ease-in-out infinite;
}}
@keyframes trash-shake {{
  0%,100% {{ transform: rotate(0deg); }}
  25% {{ transform: rotate(-8deg); }}
  75% {{ transform: rotate(8deg); }}
}}
.trash-grid {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.trash-ball {{
  position: relative; width: 2.6rem; height: 2.6rem; border-radius: 50%;
  background: #f5f5f5; border: 1px solid #e0e0e0;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; transition: all 0.3s ease;
}}
.trash-ball:hover {{
  transform: scale(1.2);
  border-color: #FF6B6B;
  background: #FFF5F5;
  box-shadow: 0 4px 12px rgba(255,107,107,0.15);
}}
.trash-num {{
  color: #888; font-size: 0.82rem; font-weight: 700; pointer-events: none;
}}
.trash-ball:hover .trash-num {{ color: #FF6B6B; }}
.trash-tooltip {{
  display: none; position: absolute; bottom: calc(100% + 10px);
  left: 50%; transform: translateX(-50%);
  background: #fff; border: 1px solid #eee;
  border-radius: 12px; padding: 10px 14px; min-width: 200px;
  z-index: 1000; box-shadow: 0 8px 24px rgba(0,0,0,0.1); color: #555;
}}
.trash-tooltip::after {{
  content: ''; position: absolute; top: 100%; left: 50%;
  transform: translateX(-50%); border: 6px solid transparent; border-top-color: #eee;
}}
.trash-ball:hover .trash-tooltip {{ display: block; }}
</style>
<details style='background:#fff;border:1px solid #eee;border-radius:20px;padding:0.8rem 1.4rem;
               margin-bottom:8px;box-shadow:0 2px 8px rgba(0,0,0,0.03)'>
  <summary style='cursor:pointer;display:flex;align-items:center;gap:0.6rem;list-style:none;
                  -webkit-appearance:none;outline:none;user-select:none'>
    <span class='trash-icon'>✕</span>
    <span style='color:#FF6B6B;font-size:0.75rem;font-weight:700;letter-spacing:2px'>號碼排除區</span>
    <span style='color:#bbb;font-size:0.68rem;margin-left:4px'>（點擊展開，懸停查看原因）</span>
  </summary>
  <div class='trash-grid' style='margin-top:0.8rem'>
    {_trash_balls}
  </div>
</details>
""", unsafe_allow_html=True)

if ml_badge:
    st.markdown(f"""
<div style='margin-top:-0.5rem;margin-bottom:1rem;padding:0.6rem 1.2rem;
     background:#FFFBF0;border:1px solid #FFE0B2;
     border-radius:12px;font-size:0.85rem'>
  {ml_badge}
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.toolbox-header {
  background: linear-gradient(135deg,#fff 0%,#f8faff 100%);
  border: 1px solid rgba(0,0,0,0.06);
  border-bottom: none;
  border-radius: 20px 20px 0 0;
  padding: 1rem 1.4rem;
  margin-top: 12px;
  position: relative;
  overflow: hidden;
  box-shadow: 0 4px 15px rgba(0,0,0,0.03);
}
.toolbox-header::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 4px;
  background: linear-gradient(90deg, #FF6B6B, #FF9671, #FFC75F, #4ECDC4, #845EC2);
}
.toolbox-title { display: flex; align-items: center; gap: 0.7rem; }
.toolbox-icon { font-size: 1.3rem; }
.toolbox-label { color: #1d1d1f; font-size: 1rem; font-weight: 900; letter-spacing: 1.5px; }
.toolbox-sub { color: #bbb; font-size: 0.68rem; margin-left: auto; letter-spacing: 1px; }
</style>
<div class='toolbox-header'>
  <div class='toolbox-title'>
    <span class='toolbox-icon'>🧰</span>
    <span class='toolbox-label'>分析工具</span>
    <span class='toolbox-sub'>v5</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── 推薦評分原因（折疊）
with st.expander("📊 查看推薦號碼評分原因"):
    st.markdown("#### — 為什麼推薦這些號碼？")

    # ── 每個號碼的推薦理由（視覺化卡片）──
    _sb_t1 = rec.score_breakdown
    _max_total_t1 = max((v["總分"] for v in _sb_t1.values()), default=1) or 1

    _dim_labels = {
        "②③遺漏+頻率": ("熱度", "近期出現頻率高"),
        "④奇偶": ("奇偶", "符合常見奇偶比"),
        "⑤大小號": ("大小", "符合常見大小比"),
        "⑥和值": ("和值", "加總落在核心區間"),
        "⑦重號/連號": ("重號", "與上期有重複號"),
        "⑧位置分析": ("位置", "常出現在該位置"),
        "⑨號碼配對": ("配對", "常與上期號同現"),
        "⑩波段週期": ("週期", "已超過平均間隔"),
        "⑪尾數分析": ("尾數", "近期高頻尾數"),
    }

    _cards_html = "<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:10px;margin:0.8rem 0'>"
    for n in rec.top7:
        b = _sb_t1[n]
        total = b["總分"]
        pct = round(total / _max_total_t1 * 100)

        # 找出該號碼得分最高的 3 個維度
        _dims = {k: b[k] for k in _dim_labels if k in b}
        _top_dims = sorted(_dims.items(), key=lambda x: x[1], reverse=True)[:3]
        _top_dims = [(k, v) for k, v in _top_dims if v > 0]

        # 標記是否在 top5
        _in5 = "border-color:#4ECDC4" if n in rec.top5 else "border-color:#eee"

        # 生成得分亮點標籤
        _tags = ""
        for dk, dv in _top_dims:
            label, tip = _dim_labels[dk]
            _tags += "<span style='display:inline-block;padding:2px 8px;margin:2px;font-size:0.7rem;background:#f0f9f8;border:1px solid #d4efed;border-radius:4px;color:#4ECDC4'>" + label + "</span>"

        if not _tags:
            _tags = "<span style='font-size:0.7rem;color:#ccc'>基礎分</span>"

        _cards_html += (
            "<div style='background:#fff;"
            "border:1px solid #eee;" + _in5 + ";border-radius:12px;padding:14px 12px;text-align:center'>"
            "  <div style='font-size:1.3rem;font-weight:700;color:#1d1d1f;font-family:Inter,sans-serif'>" + f"{n:02d}" + "</div>"
            "  <div style='margin:6px 0'>"
            "    <div style='background:#f0f0f0;border-radius:3px;height:4px;overflow:hidden'>"
            "      <div style='height:100%;width:" + str(pct) + "%;background:linear-gradient(90deg,#4ECDC4,#26A69A);border-radius:3px'></div>"
            "    </div>"
            "    <div style='font-size:0.65rem;color:#999;margin-top:3px'>" + f"{total:.1f}" + " 分</div>"
            "  </div>"
            "  <div style='margin-top:6px'>" + _tags + "</div>"
            "</div>"
        )
    _cards_html += "</div>"

    st.markdown(_cards_html, unsafe_allow_html=True)

    with st.expander("查看完整評分表"):
        rows = []
        for n in rec.top7:
            b = rec.score_breakdown[n]
            rows.append({"號碼": f"{n:02d}",
                "熱度": b["②③遺漏+頻率"],
                "奇偶": b["④奇偶"], "大小": b["⑤大小號"],
                "和值": b["⑥和值"], "重號": b["⑦重號/連號"],
                "位置": b["⑧位置分析"], "配對": b["⑨號碼配對"],
                "週期": b["⑩波段週期"], "尾數": b["⑪尾數分析"],
                "總分": b["總分"]})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with st.expander("各維度說明"):
        st.markdown("""
    | 維度 | 說明 |
    |------|------|
    | 熱度 | 指數衰減加權（半衰期50期），近期資料影響力更高 |
    | 奇偶 | 歷史最常見奇偶組合，符合加2分 |
    | 大小 | 歷史最常見大小組合，符合加2分 |
    | 和值 | 5碼加總落在歷史核心區間，加3分 |
    | 重號 | 歷史重號率>50%時加分 |
    | 位置 | 每個位置最常出現的號碼加分 |
    | 配對 | 常與上一期號碼一起出現的號碼加分 |
    | 週期 | 計算每個號碼平均出現間隔，超期加分 |
    | 尾數 | 近30期高頻尾數（個位數）對應號碼加分 |
    """)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "趨勢選號", "AI智慧選號", "尾數分析", "開獎紀錄", "我的對獎"
])

# ──────────────────────────────────────────
# Tab1：冷熱分佈
# ──────────────────────────────────────────
with tab1:
    st.markdown("#### 號碼冷熱分佈")
    n_recent = st.slider("統計最近幾期", 50, min(500, len(draws)), 100, step=50)
    recent_draws = draws[-n_recent:]
    freq_map = Counter(n for d in recent_draws for n in d.numbers)
    cycle_info = rec.stats["週期"]

    max_f = max(freq_map.values()) if freq_map else 1
    min_f = min(freq_map.get(n, 0) for n in range(1, 40))

    # 極簡熱力圖 — 只看冷熱
    sorted_freqs = sorted(freq_map.get(n, 0) for n in range(1, 40))
    hot_line = sorted_freqs[int(len(sorted_freqs) * 0.75)] if sorted_freqs else 1
    cold_line = sorted_freqs[int(len(sorted_freqs) * 0.25)] if sorted_freqs else 0

    _heat_html = (
        "<div style='text-align:center;color:#999;font-size:0.8rem;margin-bottom:10px'>"
        "近 " + str(n_recent) + " 期，紅色越深 = 出現越多"
        "</div>"
    )
    _heat_html += "<div style='display:grid;grid-template-columns:repeat(10,1fr);gap:5px;margin:0 0 0.6rem'>"
    for n in range(1, 40):
        f = freq_map.get(n, 0)

        # 三段式：熱 / 普通 / 冷
        if f >= hot_line:
            bg = "rgba(255,107,107,0.15)"
            tc = "#FF6B6B"
            fc = "#e55555"
        elif f <= cold_line:
            bg = "rgba(78,205,196,0.1)"
            tc = "#4ECDC4"
            fc = "#3bb8b0"
        else:
            bg = "#f8f8f8"
            tc = "#555"
            fc = "#999"

        _heat_html += (
            "<div style='background:" + bg + ";border-radius:8px;"
            "padding:10px 2px;text-align:center'>"
            "<div style='font-size:1.4rem;font-weight:700;color:" + tc + ";font-family:Inter,sans-serif'>" + f"{n:02d}" + "</div>"
            "<div style='font-size:0.7rem;color:" + fc + ";margin-top:2px'>" + str(f) + "次</div>"
            "</div>"
        )

    _heat_html += "<div></div>"
    _heat_html += "</div>"

    # 極簡圖例
    _heat_html += (
        "<div style='display:flex;gap:20px;justify-content:center;font-size:0.78rem;color:#FF6B6B'>"
        "<span>■ 熱號（前25%）</span>"
        "<span style='color:#4ECDC4'>■ 冷號（後25%）</span>"
        "</div>"
    )

    st.markdown(_heat_html, unsafe_allow_html=True)

    # ── 動能指標（近10期 vs 整體）──
    short_n = min(10, len(draws))
    short_draws = draws[-short_n:]
    short_freq = Counter(n for d in short_draws for n in d.numbers)

    rising, cooling = [], []
    for n in range(1, 40):
        long_rate = freq_map.get(n, 0) / n_recent
        short_rate = short_freq.get(n, 0) / short_n
        delta = short_rate - long_rate
        if long_rate > 0 and delta / long_rate > 0.5:
            rising.append((n, short_freq.get(n, 0)))
        elif long_rate > 0 and delta / long_rate < -0.4:
            cooling.append((n, short_freq.get(n, 0)))

    rising.sort(key=lambda x: -x[1])
    cooling.sort(key=lambda x: x[1])

    _mo_html = "<div style='display:flex;gap:16px;flex-wrap:wrap;margin:0.8rem 0 0.4rem'>"
    if rising:
        _mo_html += ("<div style='flex:1;min-width:180px;background:rgba(255,107,107,0.08);"
                     "border-radius:12px;padding:12px 16px'>"
                     "<div style='color:#FF6B6B;font-size:0.82rem;font-weight:700;margin-bottom:8px'>🔥 近期升溫（近10期比長期多50%以上）</div>"
                     "<div style='display:flex;flex-wrap:wrap;gap:6px'>")
        for n, cnt in rising[:8]:
            _mo_html += f"<span style='background:#FF6B6B;color:#fff;border-radius:50%;width:32px;height:32px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.85rem'>{n:02d}</span>"
        _mo_html += "</div></div>"
    if cooling:
        _mo_html += ("<div style='flex:1;min-width:180px;background:rgba(78,205,196,0.08);"
                     "border-radius:12px;padding:12px 16px'>"
                     "<div style='color:#4ECDC4;font-size:0.82rem;font-weight:700;margin-bottom:8px'>❄️ 近期降溫（近10期比長期少40%以上）</div>"
                     "<div style='display:flex;flex-wrap:wrap;gap:6px'>")
        for n, cnt in cooling[:8]:
            _mo_html += f"<span style='background:#4ECDC4;color:#fff;border-radius:50%;width:32px;height:32px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.85rem'>{n:02d}</span>"
        _mo_html += "</div></div>"
    if not rising and not cooling:
        _mo_html += "<div style='color:#999;font-size:0.85rem'>近期無明顯動能變化</div>"
    _mo_html += "</div>"
    st.markdown(_mo_html, unsafe_allow_html=True)
    st.caption(f"動能 = 近10期出現率 vs 近{n_recent}期平均出現率的偏差")

    # 配對分析
    st.markdown("---")
    st.markdown("#### 🤝 最常配對的號碼對（Top 15）")
    pairs = rec.stats["配對"][:15]
    pair_rows = [{"號碼A": f"{a:02d}", "號碼B": f"{b:02d}", "共同出現": cnt,
                  "出現率": f"{cnt/len(draws):.1%}",
                  "A在推薦": "✅" if a in rec.top7 else "", "B在推薦": "✅" if b in rec.top7 else ""}
                 for (a, b), cnt in pairs]
    st.dataframe(pd.DataFrame(pair_rows), width="stretch", hide_index=True)

    # ──────────────────────────────────────────
    # Tab3：機器學習
    # ──────────────────────────────────────────

# ──────────────────────────────────────────
# Tab2：推薦
# ──────────────────────────────────────────
with tab2:
    # ══════════════════════════════════════
    # AI 智慧選號
    # ══════════════════════════════════════
    st.markdown("---")
    st.subheader("🔗 AI 智慧選號")
    with st.expander("這組號碼怎麼選出來的？"):
        st.markdown("""
    系統會根據最近與長期的開獎紀錄，觀察：

    > 「**上期出現過哪些號碼，下一期常跟著出現哪些號碼？**」

    | 項目 | 說明 |
    |------|------|
    | 近期延續 | 看上一期與前幾期後面最常接出的號碼 |
    | 權重分配 | 越接近最近開獎，參考比重越高 |
    | 推薦邏輯 | 找出近期有連動跡象、又有機會補出的號碼 |
    | 更新速度 | 開頁後會直接重新整理，不用另外等待訓練 |
    """)

    # AI 即時計算（快取於 session_state）
    if st.session_state.get("_markov_key") != draws_key:
        with st.spinner("整理近期開獎關聯中..."):
            markov = get_markov_recommendation(draws, set(rec.killed))
        st.session_state["markov_result"] = markov
        st.session_state["_markov_key"] = draws_key
    else:
        markov = st.session_state["markov_result"]

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown("#### 🟤 精選 5 碼")
        st.markdown(num_ball(markov["top5"], "#7d3c98"), unsafe_allow_html=True)
    with mc2:
        st.markdown("#### 🟤 精選 6 碼")
        st.markdown(num_ball(markov["top6"], "#6c3483"), unsafe_allow_html=True)
    with mc3:
        st.markdown("#### 🟤 精選 7 碼")
        st.markdown(num_ball(markov["top7"], "#5b2c6f"), unsafe_allow_html=True)

    st.markdown("#### 📊 上期號碼帶動哪些熱門候選")
    trans_rows = []
    for x_str, nexts in markov["transition_top"].items():
        trans_rows.append({
            "上期號碼": x_str,
            "下期熱門候選": "　".join(f"{y}({p:.1%})" for y, p in nexts),
        })
    st.dataframe(pd.DataFrame(trans_rows), width="stretch", hide_index=True)

    st.markdown("#### AI 推薦依據（前10）")
    st.dataframe(pd.DataFrame(markov["detail"]), width="stretch", hide_index=True)


    # ──────────────────────────────────────────
    # Tab4：統計指標
    # ──────────────────────────────────────────

# ──────────────────────────────────────────
# Tab3：統計數據
# ──────────────────────────────────────────
with tab3:
    st.subheader(f"📊 統計指標（{len(draws)} 期）")
    oe = rec.stats["奇偶"]; bs = rec.stats["大小"]
    sr = rec.stats["和值"]; rp = rec.stats["重號"]; cs = rec.stats["連號"]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("最常見奇數個數", f"{oe['最常見奇數個數']} 個", f"出現機率 {oe['機率']:.1%}")
    col_b.metric("最常見大號個數", f"{bs['最常見大號個數']} 個", f"出現機率 {bs['機率']:.1%}")
    lo, hi = sr["核心區間(25-75%)"]
    col_c.metric("和值核心區間", f"{lo} ～ {hi}", f"平均和值 {sr['平均和值']}")
    col_d.metric("有重號機率", f"{rp['有重號機率']:.1%}", f"有連號機率 {cs['有連號機率']:.1%}")

    st.caption("📌 大號 = 20~39，小號 = 1~19")
    st.markdown("")

    # 奇偶比 + 大小號比
    c1, c2 = st.columns(2)
    with c1:
        labels_oe = [f"{k}奇{5-k}偶" for k in oe["分布"]]
        vals_oe   = [v*100 for v in oe["分布"].values()]
        best_oe   = f"{oe['最常見奇數個數']}奇{5-oe['最常見奇數個數']}偶"
        fig_oe = go.Figure(go.Bar(
            x=labels_oe, y=vals_oe,
            marker_color=["#FF6B6B" if l == best_oe else "#4ECDC4" for l in labels_oe],
            text=[f"{v:.1f}%" for v in vals_oe], textposition="outside",
        ))
        fig_oe.update_layout(**_dark_layout("奇偶比分布（高亮=最常見）",
            yaxis_title="出現機率 (%)", yaxis=dict(range=[0, max(vals_oe)*1.3])))
        st.plotly_chart(fig_oe, width="stretch")
        st.caption(f"👉 下一期最可能是 **{best_oe}**（歷史機率 {oe['機率']:.1%}）")

    with c2:
        labels_bs = [f"{k}大{5-k}小" for k in bs["分布"]]
        vals_bs   = [v*100 for v in bs["分布"].values()]
        best_bs   = f"{bs['最常見大號個數']}大{5-bs['最常見大號個數']}小"
        fig_bs = go.Figure(go.Bar(
            x=labels_bs, y=vals_bs,
            marker_color=["#FF6B6B" if l == best_bs else "#4ECDC4" for l in labels_bs],
            text=[f"{v:.1f}%" for v in vals_bs], textposition="outside",
        ))
        fig_bs.update_layout(**_dark_layout("大小號比分布（高亮=最常見）",
            yaxis_title="出現機率 (%)", yaxis=dict(range=[0, max(vals_bs)*1.3])))
        st.plotly_chart(fig_bs, width="stretch")
        st.caption(f"👉 下一期最可能是 **{best_bs}**（歷史機率 {bs['機率']:.1%}）")

    # 和值分布
    st.markdown("---")
    sums_all = [sum(d.numbers) for d in draws]
    fig_sum = go.Figure(go.Histogram(x=sums_all, nbinsx=40, marker_color="#845EC2", opacity=0.8))
    fig_sum.add_vrect(x0=lo, x1=hi, fillcolor="#FF6B6B", opacity=0.1,
                      annotation_text=f"核心區間 {lo}~{hi}", annotation_position="top left")
    fig_sum.update_layout(**_dark_layout(
        f"5碼和值分布（平均={sr['平均和值']}，核心區間={lo}~{hi}）",
        height=280, xaxis_title="5碼加總", yaxis_title="出現次數"))
    st.plotly_chart(fig_sum, width="stretch")
    st.caption("👉 選號時讓5碼加總落在紅色區間內，符合歷史規律")

    # 殺號成功率（快取）
    st.markdown("---")
    st.markdown("#### ⚔️ 殺號公式成功率（理論基準 66.4%）")
    if st.session_state.get("_bt_key") != draws_key:
        bt_results = sorted([backtest(draws, n, fn) for n, fn in FORMULAS.items()],
                            key=lambda r: -r.success_rate)
        st.session_state["_bt_results"] = bt_results
        st.session_state["_bt_key"] = draws_key
    else:
        bt_results = st.session_state["_bt_results"]

    rates = [r.success_rate*100 for r in bt_results]
    names = [r.formula_name for r in bt_results]
    fig_kl = go.Figure(go.Bar(
        x=names, y=rates,
        marker_color=["#FF6B6B" if v >= 66.4 else "#4ECDC4" for v in rates],
        text=[f"{v:.1f}%" for v in rates], textposition="outside",
    ))
    fig_kl.add_hline(y=66.4, line_dash="dash", line_color="#999",
                     annotation_text="理論基準 66.4%", annotation_position="top right")
    fig_kl.update_layout(**_dark_layout("各公式殺號成功率（紅=超越基準）",
        yaxis_title="成功率 (%)", yaxis=dict(range=[40, max(rates)*1.2]),
        margin=dict(t=50, b=60, l=40, r=20)))
    st.plotly_chart(fig_kl, width="stretch")

    # ── 演算法 PK 擂台 ──
    st.markdown("---")
    st.markdown("#### 🏅 演算法 PK 擂台（各系統推薦共同號碼數）")
    st.caption("統計各系統 Top5 推薦號碼與最近10期實際開獎的重疊度（需先跑 ML & LSTM）")

    _pk_n = min(10, len(draws))
    _pk_draws = draws[-_pk_n:]
    _pk_actual = [set(d.numbers) for d in _pk_draws]

    def _pk_score(top5_nums):
        return sum(len(set(top5_nums) & a) for a in _pk_actual) / _pk_n

    _pk_rows = []
    _pk_stat = _pk_score(rec.top5)
    _pk_rows.append({"演算法": "統計法", "近10期平均命中": f"{_pk_stat:.2f}", "_score": _pk_stat})

    if "ml_result" in st.session_state:
        _s = _pk_score(st.session_state["ml_result"]["top5"])
        _pk_rows.append({"演算法": "XGBoost ML", "近10期平均命中": f"{_s:.2f}", "_score": _s})
    if "lstm_result" in st.session_state:
        _s = _pk_score(st.session_state["lstm_result"]["top5"])
        _pk_rows.append({"演算法": "LSTM+Attention", "近10期平均命中": f"{_s:.2f}", "_score": _s})
    if "markov_result" in st.session_state:
        _s = _pk_score(st.session_state["markov_result"]["top5"])
        _pk_rows.append({"演算法": "AI智慧推薦", "近10期平均命中": f"{_s:.2f}", "_score": _s})
    if "quad_top5" in st.session_state:
        _s = _pk_score(st.session_state["quad_top5"])
        _pk_rows.append({"演算法": "四方集成", "近10期平均命中": f"{_s:.2f}", "_score": _s})

    if _pk_rows:
        _pk_rows.sort(key=lambda x: -x["_score"])
        _pk_best = _pk_rows[0]["演算法"]
        _pk_scores = [r["_score"] for r in _pk_rows]
        _pk_names  = [r["演算法"] for r in _pk_rows]
        fig_pk = go.Figure(go.Bar(
            x=_pk_names, y=_pk_scores,
            marker_color=["#FF6B6B" if n == _pk_best else "#4ECDC4" for n in _pk_names],
            text=[f"{v:.2f}" for v in _pk_scores], textposition="outside",
        ))
        fig_pk.update_layout(**_dark_layout(
            f"近10期平均每次命中碼數（紅=本週最強）", height=280,
            yaxis_title="平均命中碼數 / 5",
            yaxis=dict(range=[0, max(_pk_scores)*1.4 if _pk_scores else 1]),
        ))
        st.plotly_chart(fig_pk, width="stretch")
        st.success(f"🥇 本週最強演算法：**{_pk_best}**（近10期平均命中 {_pk_rows[0]['_score']:.2f} 碼）")
        st.caption("※ 其他推薦結果需先在 AI智慧選號頁載入過才會出現")

    # ──────────────────────────────────────────
    # Tab5：歷史開獎紀錄 (RWD 卡片化)
    # ──────────────────────────────────────────

# ──────────────────────────────────────────
# Tab4：歷史開獎
# ──────────────────────────────────────────
with tab4:
    st.subheader("📅 歷史開獎紀錄")
    show_n = st.slider("顯示最近幾期", 20, min(500, len(draws)), 50, step=10)

    st.markdown("""
    <style>
    .history-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 12px;
    margin-top: 1rem;
    }
    .history-card {
    background: #fff;
    border: 1px solid #eee;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    transition: transform 0.2s ease;
    }
    .history-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    border-color: #4ECDC4;
    }
    .hist-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #f0f0f0;
    padding-bottom: 8px;
    margin-bottom: 12px;
    }
    .hist-period { font-weight: 800; color: #1d1d1f; font-size: 1.05rem; letter-spacing: 1px; }
    .hist-tags { display: flex; gap: 6px; }
    .hist-tag { font-size: 0.7rem; padding: 2px 8px; border-radius: 99px; font-weight: 600; }
    .tag-sum { background: #f0f0f0; color: #555; }
    .tag-odd { background: #FFF5F5; color: #FF6B6B; }
    .tag-big { background: #F0FAF9; color: #4ECDC4; }
    .hist-balls { display: flex; gap: 8px; justify-content: space-between; }
    .h-ball {
    width: 2.4rem; height: 2.4rem; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 1rem;
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    color: #333; border: 1px solid #dee2e6;
    }
    @media (max-width: 480px) {
    .history-grid { grid-template-columns: 1fr; }
    .h-ball { width: 2.2rem; height: 2.2rem; font-size: 0.95rem; }
    }
    </style>
    """, unsafe_allow_html=True)

    _cards_html = "<div class='history-grid'>"
    for d in reversed(draws[-show_n:]):
        nums = d.numbers
        s_sum = sum(nums)
        s_odd = sum(1 for n in nums if n % 2 == 1)
        s_big = sum(1 for n in nums if n >= 20)
        balls_html = "".join([f"<div class='h-ball'>{n:02d}</div>" for n in nums])
        _cards_html += f"""
    <div class="history-card">
      <div class="hist-header">
    <div class="hist-period">{d.period}</div>
    <div class="hist-tags">
      <span class="hist-tag tag-sum">和 {s_sum}</span>
      <span class="hist-tag tag-odd">{s_odd}奇</span>
      <span class="hist-tag tag-big">{s_big}大</span>
    </div>
      </div>
      <div class="hist-balls">{balls_html}</div>
    </div>"""
    _cards_html += "</div>"
    st.markdown(_cards_html, unsafe_allow_html=True)


# ──────────────────────────────────────────
# Tab5：對獎驗證
# ──────────────────────────────────────────
with tab5:
    # ══ 對獎驗證 ══
    st.markdown("""
    <div style='background:#fff;border:1px solid #eee;border-radius:16px;padding:1.8rem 1.5rem 1.2rem;margin:1rem 0;
     box-shadow:0 2px 8px rgba(0,0,0,0.04);position:relative;overflow:hidden'>
      <div style='position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#FF6B6B,#4ECDC4,#845EC2)'></div>
      <div style='text-align:center;margin-bottom:0.5rem'>
    <div style='font-size:1.4rem;margin-bottom:6px'>🎰</div>
    <div style='color:#1d1d1f;font-size:1.2rem;font-weight:900;letter-spacing:3px;'>我的對獎</div>
    <div style='color:#999;font-size:0.95rem;margin-top:6px;letter-spacing:1px'>輸入你的號碼 → 掃描歷史 → 算出中獎率</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='color:#666;font-size:1rem;margin-bottom:4px'>輸入 5 個號碼（空格、逗號、頓號皆可）例如：<b>5 12 19 27 34</b></div>", unsafe_allow_html=True)
    _vc1, _vc2 = st.columns([3, 1])
    with _vc1:
        custom_input = st.text_input("輸入 5 個號碼", placeholder="5 12 19 27 34",
                                      label_visibility="collapsed", key="validator_input")
    with _vc2:
        verify_btn = st.button("開始驗證", type="primary", use_container_width=True)

    if custom_input and verify_btn:
        try:
            import re
            _raw = re.split(r'[,，\s、/\-]+', custom_input.strip())
            _all_parsed = [int(x) for x in _raw if x.strip() and x.strip().isdigit()]
            _valid = sorted(set(n for n in _all_parsed if 1 <= n <= 39))[:5]
            if len(_valid) < 5 and len(_all_parsed) > 0:
                _invalid = [n for n in _all_parsed if not (1 <= n <= 39)]
                if _invalid:
                    st.warning(f"已自動移除無效號碼：{_invalid}（須介於 1~39）")
            custom_nums = _valid
            if len(custom_nums) == 5:
                killed_set = set(rec.killed)
                blocked = [n for n in custom_nums if n in killed_set]
                custom_sum = sum(custom_nums)
                custom_odd = sum(1 for n in custom_nums if n % 2 == 1)
                custom_big = sum(1 for n in custom_nums if n >= 20)
                lo, hi = rec.stats["和值"]["核心區間(25-75%)"]
                sum_ok = lo <= custom_sum <= hi

                # 歷史模擬動畫（加速→減速）
                import time
                _slot_ph = st.empty()
                _total_frames = 24
                for _frame in range(_total_frames):
                    _rand_nums = sorted(np.random.choice(range(1, 40), 5, replace=False))
                    _progress = round((_frame + 1) / _total_frames * 100)
                    _scan_period = round((_frame + 1) / _total_frames * len(draws))
                    _slot_ph.markdown(f"""
    <div style='text-align:center;padding:1.2rem;background:#f8f9fa;
            border-radius:16px;margin:0.5rem 0;border:1px solid #eee'>
      <div style='display:flex;justify-content:center;gap:14px'>
    {"".join(f"<span style='display:inline-flex;width:3.2rem;height:3.2rem;border-radius:50%;background:linear-gradient(135deg,#4ECDC4,#26A69A);align-items:center;justify-content:center;font-weight:900;font-size:1.1rem;color:#fff;box-shadow:0 4px 12px rgba(78,205,196,0.3)'>{n:02d}</span>" for n in _rand_nums)}
      </div>
      <div style='margin-top:12px'>
    <div style='background:#e0e0e0;border-radius:99px;height:6px;overflow:hidden;width:60%;margin:0 auto'>
      <div style='height:100%;width:{_progress}%;background:linear-gradient(90deg,#4ECDC4,#26A69A);border-radius:99px;transition:width 0.1s'></div>
    </div>
    <div style='color:#4ECDC4;font-size:0.75rem;margin-top:8px;font-weight:600'>🔄 掃描第 {_scan_period} / {len(draws)} 期...</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
                    # 加速→減速：前半快、後半慢
                    if _frame < _total_frames // 3:
                        time.sleep(0.05)
                    elif _frame < _total_frames * 2 // 3:
                        time.sleep(0.1)
                    else:
                        time.sleep(0.18)

                # 最終結果
                _slot_ph.markdown(f"""
    <div style='text-align:center;padding:1.2rem;background:#f8f9fa;border-radius:16px;margin:0.5rem 0;
     border:1px solid #eee'>
      <div style='display:flex;justify-content:center;gap:12px'>
    {"".join(f"<span style='display:inline-flex;width:3.2rem;height:3.2rem;border-radius:50%;background:linear-gradient(135deg,{'#FF6B6B,#D32F2F' if n in killed_set else '#4ECDC4,#26A69A' if n in set(rec.top7) else '#90CAF9,#42A5F5'});align-items:center;justify-content:center;font-weight:900;font-size:1.1rem;color:#fff;box-shadow:0 4px 12px rgba(0,0,0,0.1)'>{n:02d}</span>" for n in custom_nums)}
      </div>
      <div style='color:#999;font-size:0.7rem;margin-top:8px'>🟢 推薦號碼　🔴 排除號碼　🔵 普通號碼</div>
    </div>
    """, unsafe_allow_html=True)

                # 歷史中獎率計算
                hit3 = sum(1 for d in draws if len(set(custom_nums) & set(d.numbers)) >= 3)
                hit4 = sum(1 for d in draws if len(set(custom_nums) & set(d.numbers)) >= 4)
                hit5 = sum(1 for d in draws if len(set(custom_nums) & set(d.numbers)) >= 5)
                hit3_pct = round(hit3 / len(draws) * 100, 2)
                hit4_pct = round(hit4 / len(draws) * 100, 2)
                hit5_pct = round(hit5 / len(draws) * 100, 2)

                # 大儀表板顯示中獎率
                if hit3_pct >= 5:
                    _gauge_color = "#4ECDC4"
                    _gauge_text = "🏆 歷史表現優秀！"
                    _gauge_bg = "#f0faf9"
                elif hit3_pct >= 3:
                    _gauge_color = "#FF9671"
                    _gauge_text = "⚡ 有潛力的組合"
                    _gauge_bg = "#fff8f5"
                elif hit3_pct >= 1:
                    _gauge_color = "#42A5F5"
                    _gauge_text = " 正常機率範圍"
                    _gauge_bg = "#f5f9ff"
                else:
                    _gauge_color = "#845EC2"
                    _gauge_text = "✨ 冷門組合，搏一把"
                    _gauge_bg = "#f8f5ff"
                _gauge_deg = min(round(hit3_pct / 10 * 360), 360)
                # 用 CSS conic-gradient 代替 SVG
                _ring_html = (
                    "<div style='position:relative;width:160px;height:160px;margin:0 auto 20px'>"
                    "<div style='width:100%;height:100%;border-radius:50%;"
                    "background:conic-gradient(" + _gauge_color + " 0deg," + _gauge_color + " " + str(_gauge_deg) + "deg,#f0f0f0 " + str(_gauge_deg) + "deg,#f0f0f0 360deg);"
                    "-webkit-mask:radial-gradient(farthest-side,transparent calc(100% - 14px),#000 calc(100% - 13px));"
                    "mask:radial-gradient(farthest-side,transparent calc(100% - 14px),#000 calc(100% - 13px))'></div>"
                    "<div style='position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center'>"
                    "<div style='font-size:2.8rem;font-weight:900;color:" + _gauge_color + ";line-height:1'>" + str(hit3_pct) + "</div>"
                    "<div style='font-size:0.75rem;color:#999;margin-top:4px;letter-spacing:1px'>%</div>"
                    "</div></div>"
                )
                # 5碼全中特效
                _hit5_glow = "background:#fff8f0;border:1px solid #FFE0B2;box-shadow:0 4px 12px rgba(255,152,0,0.1)" if hit5 > 0 else "background:#f8f9fa;border:1px solid #eee"
                _hit5_icon = "🎉 " if hit5 > 0 else ""
                _result_html = (
                    "<div style='text-align:center;margin:1rem 0;padding:2rem 1.5rem;background:" + _gauge_bg + ";"
                    "border-radius:20px;position:relative;overflow:hidden;border:1px solid #eee'>"
                    "  <div style='position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#FF6B6B,#4ECDC4,#845EC2)'></div>"
                    "  <div style='font-size:0.75rem;color:#999;letter-spacing:4px;margin-bottom:20px;font-weight:700'>歷 史 模 擬 中 獎 率</div>"
                    + _ring_html +
                    "  <div style='font-size:1rem;color:#1d1d1f;font-weight:800;letter-spacing:1px'>" + _gauge_text + "</div>"
                    "  <div style='font-size:0.72rem;color:#999;margin-top:6px'>" + str(hit3) + " / " + str(len(draws)) + " 期命中 3 碼以上</div>"
                    "  <div style='display:flex;justify-content:center;gap:1rem;margin-top:1.5rem'>"
                    "    <div style='" + _hit5_glow + ";border-radius:14px;padding:14px 22px;min-width:80px;text-align:center'>"
                    "      <div style='font-size:1.8rem;font-weight:900;color:" + ("#FF9671" if hit5 > 0 else "#ccc") + "'>" + _hit5_icon + str(hit5) + "</div>"
                    "      <div style='font-size:0.62rem;color:#999;margin-top:4px;letter-spacing:1px'>5碼全中</div>"
                    "    </div>"
                    "    <div style='background:#f8f9fa;border:1px solid #eee;border-radius:14px;padding:14px 22px;min-width:80px;text-align:center'>"
                    "      <div style='font-size:1.8rem;font-weight:900;color:#1d1d1f'>" + str(hit4) + "</div>"
                    "      <div style='font-size:0.62rem;color:#999;margin-top:4px;letter-spacing:1px'>中4碼</div>"
                    "    </div>"
                    "    <div style='background:#f8f9fa;border:1px solid #eee;border-radius:14px;padding:14px 22px;min-width:80px;text-align:center'>"
                    "      <div style='font-size:1.8rem;font-weight:900;color:#1d1d1f'>" + str(hit3) + "</div>"
                    "      <div style='font-size:0.62rem;color:#999;margin-top:4px;letter-spacing:1px'>中3碼</div>"
                    "    </div>"
                    "  </div>"
                    "</div>"
                )
                st.markdown(_result_html, unsafe_allow_html=True)

                # 四項統計指標
                col_v1, col_v2, col_v3, col_v4 = st.columns(4)
                col_v1.metric("和值", custom_sum, f"{'✅ 核心區間' if sum_ok else '⚠️ 偏離'}")
                col_v2.metric("奇數", f"{custom_odd} 個", f"常見:{rec.stats['奇偶']['最常見奇數個數']}個")
                col_v3.metric("大號", f"{custom_big} 個", f"常見:{rec.stats['大小']['最常見大號個數']}個")
                col_v4.metric("殺號衝突", len(blocked), f"{'⚠️ '+' '.join(str(n) for n in blocked) if blocked else '✅ 無'}")

                if blocked:
                    st.error(f"⚠️ 號碼 {', '.join(str(n) for n in blocked)} 在殺號組內，建議替換")
            else:
                st.error(f"需要 5 個號碼，目前有效號碼只有 {len(custom_nums)} 個，請補齊")
        except Exception:
            st.error("格式錯誤，請輸入 5 個數字（例如：5 12 19 27 34 或 5,12,19,27,34）")



st.caption("⚠️ 彩票為隨機事件，本工具僅供統計參考，請理性投注。")
