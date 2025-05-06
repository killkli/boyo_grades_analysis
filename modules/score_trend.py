import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def score_trend_tab(df, current_context_info, get_gemini_advice):
    st.subheader("成績分布與趨勢")
    if '成績' in df.columns:
        desc = df['成績'].describe().to_dict()
        quantiles = df['成績'].quantile([0.25, 0.5, 0.75]).to_dict()
        stats = {**desc, '分位數': quantiles}
        st.write(df['成績'].describe())
        st.write("#### 成績分布直方圖 (靜態)")
        fig, ax = plt.subplots()
        sns.histplot(df['成績'], kde=True, ax=ax)
        st.pyplot(fig)
        st.write("#### 成績分布直方圖 (互動式)")
        fig_px = px.histogram(df, x='成績', nbins=20, title='成績分布直方圖', marginal="box", histnorm=None)
        st.plotly_chart(fig_px, use_container_width=True)
        with st.spinner('正在產生AI分析建議，請稍候...'):
            st.info(get_gemini_advice('成績分布與趨勢', stats=stats, context_info=current_context_info))
    else:
        st.warning("本工作表無 '成績' 欄位") 