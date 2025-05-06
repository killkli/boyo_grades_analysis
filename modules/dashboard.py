import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def dashboard_tab(df):
    """
    Dashboard 總覽分頁：顯示學生人數、平均分數、及格率等指標，並以互動直方圖呈現成績分布。
    
    參數：
        df: pd.DataFrame，成績資料表，需包含 '姓名'、'成績' 欄位
    回傳：
        無（直接於 Streamlit 畫面顯示）
    """
    st.header("Dashboard 總覽")
    col1, col2, col3 = st.columns(3)
    if '姓名' in df.columns:
        # 顯示學生人數
        col1.metric("學生人數", df['姓名'].nunique())
    if '成績' in df.columns:
        # 顯示平均分數與及格率
        col2.metric("平均分數", f"{df['成績'].mean():.2f}")
        col3.metric("及格率", f"{(df['成績']>=60).mean()*100:.1f}%")
        # 互動成績分布直方圖（hover 顯示分數區間與人數）
        fig = px.histogram(df, x='成績', nbins=20, title='成績分布直方圖')
        fig.update_traces(hovertemplate='分數區間: %{x}<br>人數: %{y}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("本資料無成績欄位") 