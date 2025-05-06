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
    # 讀取所有工作表名稱
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    st.sidebar.header("選擇分析功能")
    sheet = st.sidebar.selectbox("選擇工作表", sheet_names)
    df = pd.read_excel(xls, sheet_name=sheet)
    st.write(f"### 預覽：{sheet}")
    st.dataframe(df.head(20))
    
    analysis_type = st.sidebar.selectbox(
        "選擇分析類型",
        [
            "通過率分析",
            "成績分布與趨勢",
            "進階檢測與晉級分析",
            "檢測參與度",
            "時間序列分析",
            "個案追蹤",
            "交叉分析",
            "成績分群分析"
        ]
    )
    
    if analysis_type == "通過率分析":
        st.subheader("通過率分析")
        if '是否通過？' in df.columns:
            pass_rate = df['是否通過？'].value_counts(normalize=True) * 100
            st.write(pass_rate)
            fig, ax = plt.subplots()
            pass_rate.plot(kind='bar', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("本工作表無 '是否通過？' 欄位")
    elif analysis_type == "成績分布與趨勢":
        st.subheader("成績分布與趨勢")
        if '成績' in df.columns:
            st.write(df['成績'].describe())
            fig, ax = plt.subplots()
            sns.histplot(df['成績'], kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("本工作表無 '成績' 欄位")
    elif analysis_type == "進階檢測與晉級分析":
        st.subheader("進階檢測與晉級分析")
        if '是否計算為進階檢測？' in df.columns:
            st.write(df['是否計算為進階檢測？'].value_counts())
        else:
            st.warning("本工作表無 '是否計算為進階檢測？' 欄位")
    elif analysis_type == "檢測參與度":
        st.subheader("檢測參與度")
        if '姓名' in df.columns:
            st.write(df['姓名'].value_counts())
        else:
            st.warning("本工作表無 '姓名' 欄位")
    elif analysis_type == "時間序列分析":
        st.subheader("時間序列分析")
        if '日期' in df.columns and '成績' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
            st.line_chart(df.sort_values('日期').set_index('日期')['成績'])
        else:
            st.warning("本工作表無 '日期' 或 '成績' 欄位")
    elif analysis_type == "個案追蹤":
        st.subheader("個案追蹤")
        if '姓名' in df.columns and '成績' in df.columns:
            student = st.selectbox("選擇學生", df['姓名'].unique())
            student_df = df[df['姓名'] == student]
            st.write(student_df[['日期', '檢測名稱', '成績', '是否通過？']])
            if '日期' in student_df.columns:
                student_df['日期'] = pd.to_datetime(student_df['日期'], errors='coerce')
                st.line_chart(student_df.sort_values('日期').set_index('日期')['成績'])
        else:
            st.warning("本工作表無 '姓名' 或 '成績' 欄位")
    elif analysis_type == "交叉分析":
        st.subheader("交叉分析")
        if '是否通過？' in df.columns and '學生考核類別' in df.columns:
            cross = pd.crosstab(df['學生考核類別'], df['是否通過？'])
            st.write(cross)
            st.bar_chart(cross)
        else:
            st.warning("本工作表無 '是否通過？' 或 '學生考核類別' 欄位")
    elif analysis_type == "成績分群分析":
        st.subheader("成績分群分析（KMeans）")
        if '成績' in df.columns:
            n_clusters = st.slider("選擇分群數量", min_value=2, max_value=8, value=3)
            valid_scores = df['成績'].dropna().values.reshape(-1, 1)
            if len(valid_scores) >= n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(valid_scores)
                df_clustered = df.copy()
                df_clustered.loc[df['成績'].notna(), '分群'] = clusters
                st.write(df_clustered[['姓名', '成績', '分群']].sort_values('分群'))
                fig, ax = plt.subplots()
                sns.histplot(data=df_clustered, x='成績', hue='分群', multiple='stack', palette='tab10', ax=ax)
                st.pyplot(fig)
            else:
                st.warning("有效成績數量不足以分成所選群數")
        else:
            st.warning("本工作表無 '成績' 欄位")
else:
    st.info("請先上傳Excel檔案") 