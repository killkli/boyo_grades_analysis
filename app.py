import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from io import BytesIO
from sklearn.cluster import KMeans
import numpy as np
import matplotlib
import platform
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression
import random
import os
import google.generativeai as genai
import plotly.express as px
from modules.sidebar import sidebar_filters
from modules.dashboard import dashboard_tab
from modules.score_distribution import score_distribution_tab
from modules.pass_rate import pass_rate_tab
from modules.score_trend import score_trend_tab
from modules.advanced_test import advanced_test_tab
from modules.participation import participation_tab
from modules.time_series import time_series_tab
from modules.case_tracking import case_tracking_tab
from modules.cross_analysis import cross_analysis_tab
from modules.score_clustering import score_clustering_tab
from modules.auto_test_clustering import auto_test_clustering_tab
from modules.correlation import correlation_tab
from modules.prediction import prediction_tab


def get_gemini_advice(context, stats=None, group=None, context_info=None):
    """
    呼叫 Gemini API 產生分析建議，若無 API 金鑰則 fallback 為 mock。
    context: 分析區塊主題
    stats: 相關統計數據（dict）
    group: 分群標籤或學生名單
    context_info: dict, e.g. {'學校': [...], '年級': [...], '檢測名稱': [...]}
    """
    api_key = os.environ.get('GEMINI_API_KEY')
    prompt = ""
    info_str = ""
    if context_info:
        info_str = "分析對象資訊："
        for k, v in context_info.items():
            if v:
                info_str += f"{k}：{', '.join(map(str, v))}；"
        if info_str:
            info_str += "\n"
    if context == '分數分布分析':
        avg = stats.get('avg', 0)
        std = stats.get('std', 0)
        pass_rate = stats.get('pass_rate', 0)
        prompt = f"{info_str}請根據以下數據，產生一段專業且具教育意義的成績分布分析建議，並給出指標說明與解讀建議：\n平均分數：{avg:.2f}，標準差：{std:.2f}，及格率：{pass_rate:.1f}%。"
    elif context == '分群分析':
        if group is not None and isinstance(group, dict):
            group_str = '\n'.join(
                [f"群組 {k}: 平均 {v['mean']:.1f}，及格率 {v['pass_rate']:.1f}%" for k, v in group.items()])
            prompt = f"{info_str}請根據以下分群摘要，產生一段專業且具教育意義的分群分析建議，針對不同分群給予個人化學習建議：\n{group_str}"
        else:
            prompt = f"{info_str}請產生一段分群分析的教育建議。"
    elif context == '預測分析':
        if group is not None and isinstance(group, list):
            if not group:
                return "無高風險學生。"
            group_str = '\n'.join(
                [f"{stu['姓名']}：預測分數 {stu['預測分數']:.1f}" for stu in group])
            prompt = f"{info_str}以下為高風險學生名單，請針對每位學生給予個人化學習建議：\n{group_str}"
        else:
            prompt = f"{info_str}請產生一段預測分析的教育建議。"
    else:
        prompt = f"{info_str}請針對本區塊分析數據，產生一段教育意義建議。"
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name='gemini-2.0-flash')
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"[Gemini API 錯誤 fallback] {str(e)}\n{mock_gemini_advice(context, stats, group, context_info)}"
    else:
        return mock_gemini_advice(context, stats, group, context_info)


def mock_gemini_advice(context, stats=None, group=None, context_info=None):
    info_str = ""
    if context_info:
        info_str = "分析對象資訊："
        for k, v in context_info.items():
            if v:
                info_str += f"{k}：{', '.join(map(str, v))}；"
        if info_str:
            info_str += "\n"
    if context == '分數分布分析':
        avg = stats.get('avg', 0)
        std = stats.get('std', 0)
        pass_rate = stats.get('pass_rate', 0)
        advice = f"{info_str}平均分數為 {avg:.2f}，標準差 {std:.2f}，及格率 {pass_rate:.1f}%。\n"
        if avg >= 80:
            advice += "整體表現優異，可持續精進。"
        elif avg >= 60:
            advice += "大多數學生已達及格，建議針對落後學生加強輔導。"
        else:
            advice += "平均分數偏低，建議檢視教學內容與學習策略。"
        if std > 15:
            advice += "\n分數分布離散，學生間學習落差大。"
        return advice
    elif context == '分群分析':
        advice = info_str if info_str else ""
        if group is not None and isinstance(group, dict):
            advice += "分群摘要：\n"
            for k, v in group.items():
                advice += f"群組 {k}: 平均 {v['mean']:.1f}，及格率 {v['pass_rate']:.1f}%\n"
                if v['mean'] < 60:
                    advice += "→ 建議加強基礎教學。\n"
                elif v['mean'] > 80:
                    advice += "→ 可安排進階學習。\n"
            return advice
        return advice + "分群有助於辨識不同學習層次學生，建議針對低分群給予補救教學。"
    elif context == '預測分析':
        advice = info_str if info_str else ""
        if group is not None and isinstance(group, list):
            if not group:
                return advice + "無高風險學生。"
            advice += "高風險學生建議：\n"
            for stu in group:
                advice += f"{stu['姓名']}：預測分數 {stu['預測分數']:.1f}，建議加強輔導。\n"
            return advice
        return advice + "預測分析可協助及早發現學習風險，建議針對預測分數低於及格線學生進行輔導。"
    return info_str + "本區塊分析有助於掌握學生學習狀況，建議依據數據調整教學策略。"

def get_available_font(font_list):
    available = set(f.name for f in fm.fontManager.ttflist)
    for font in font_list:
        if font in available:
            return font
    return None

system = platform.system()
if system == 'Windows':
    font_candidates = ['Microsoft JhengHei', 'DFKai-SB', 'SimHei', 'SimSun']
elif system == 'Darwin':  # macOS
    font_candidates = ['PingFang TC', 'Heiti TC', '黑體-繁', 'STHeiti', 'Bpmf GenYo Gothic', 'Bpmf GenSeki Gothic', 'Bpmf Zihi Sans']
else:  # Linux 或其他
    font_candidates = ['Noto Sans TC', 'AR PL UMing TW', 'AR PL UKai TW', 'Source Han Serif TC', 'Source Han Sans TC']

font_name = get_available_font(font_candidates)
if font_name:
    matplotlib.rc('font', family=font_name)
else:
    matplotlib.rc('font', family='Heiti TC')  # fallback
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="成績分析App", layout="wide")
st.title("學生成績Excel分析App")

uploaded_file = st.file_uploader("請上傳成績Excel檔案", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    sheet = st.sidebar.selectbox("選擇工作表", sheet_names)
    df = pd.read_excel(xls, sheet_name=sheet)
    st.write(f"### 預覽：{sheet}")
    st.dataframe(df.head(20))

    # ===== Sidebar 篩選區模組化 =====
    df, current_context_info = sidebar_filters(df)

    # ===== 分頁式分析（Tabs）與首頁 Dashboard =====
    tab_names = [
        "Dashboard", "分數分布分析", "通過率分析", "成績分布與趨勢", "進階檢測與晉級分析", "檢測參與度", "時間序列分析", "個案追蹤", "交叉分析", "成績分群分析", "自動分群檢測分布分析", "相關性分析", "預測分析"
    ]
    tabs = st.tabs(tab_names)

    # Dashboard
    with tabs[0]:
        dashboard_tab(df)

    # 分數分布分析
    with tabs[1]:
        score_distribution_tab(df, current_context_info, get_gemini_advice)

    # 其餘分析分頁
    for idx, tab_name in enumerate(tab_names):
        if tab_name in ["Dashboard", "分數分布分析", "相關性分析", "預測分析"]:
            continue
        with tabs[idx]:
            if tab_name == "通過率分析":
                pass_rate_tab(df, current_context_info, get_gemini_advice)
            elif tab_name == "成績分布與趨勢":
                score_trend_tab(df, current_context_info, get_gemini_advice)
            elif tab_name == "進階檢測與晉級分析":
                advanced_test_tab(df, current_context_info, get_gemini_advice)
            elif tab_name == "檢測參與度":
                participation_tab(df, current_context_info, get_gemini_advice)
            elif tab_name == "時間序列分析":
                time_series_tab(df, current_context_info, get_gemini_advice)
            elif tab_name == "個案追蹤":
                case_tracking_tab(df, current_context_info, get_gemini_advice)
            elif tab_name == "交叉分析":
                cross_analysis_tab(df, current_context_info, get_gemini_advice)
            elif tab_name == "成績分群分析":
                score_clustering_tab(df, current_context_info, get_gemini_advice)
            elif tab_name == "自動分群檢測分布分析":
                auto_test_clustering_tab(df, current_context_info, get_gemini_advice)

    # 相關性分析
    with tabs[-2]:
        correlation_tab(df)

    # 預測分析
    with tabs[-1]:
        prediction_tab(df, current_context_info, get_gemini_advice)
else:
    st.info("請先上傳Excel檔案")

