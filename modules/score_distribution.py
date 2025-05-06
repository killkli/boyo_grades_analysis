import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def score_distribution_tab(df, current_context_info, get_gemini_advice):
    st.header("分數分布分析")
    if '成績' in df.columns:
        avg = df['成績'].mean()
        std = df['成績'].std()
        pass_rate = (df['成績'] >= 60).mean() * 100
        st.metric("平均分數", f"{avg:.2f}")
        st.metric("標準差", f"{std:.2f}")
        st.metric("及格率", f"{pass_rate:.1f}%")
        st.write("#### 分數分布直方圖 (靜態)")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['成績'], kde=True, ax=ax1)
        st.pyplot(fig1)
        st.write("#### 分數分布直方圖 (互動式)")
        fig1_px = px.histogram(df, x='成績', nbins=20, title='分數分布直方圖', marginal="box", histnorm=None)
        st.plotly_chart(fig1_px, use_container_width=True)
        st.write("#### 分數分布箱型圖 (靜態)")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x=df['成績'], ax=ax2)
        st.pyplot(fig2)
        st.write("#### 分數分布箱型圖 (互動式)")
        fig2_px = px.box(df, y='成績', title='分數分布箱型圖')
        st.plotly_chart(fig2_px, use_container_width=True)
        st.info(get_gemini_advice('分數分布分析', stats={'avg': avg, 'std': std, 'pass_rate': pass_rate}, context_info=current_context_info))
    else:
        st.warning("本工作表無 '成績' 欄位") 