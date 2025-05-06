import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def dashboard_tab(df):
    st.header("Dashboard 總覽")
    col1, col2, col3 = st.columns(3)
    if '姓名' in df.columns:
        col1.metric("學生人數", df['姓名'].nunique())
    if '成績' in df.columns:
        col2.metric("平均分數", f"{df['成績'].mean():.2f}")
        col3.metric("及格率", f"{(df['成績']>=60).mean()*100:.1f}%")
        fig, ax = plt.subplots()
        sns.histplot(df['成績'], kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.info("本資料無成績欄位") 