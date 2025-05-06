import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def score_trend_tab(df, current_context_info, get_gemini_advice):
    """
    成績分布與趨勢分頁：顯示成績描述統計、分布直方圖（含互動式），並呼叫 Gemini API 產生分析建議。
    
    參數：
        df: pd.DataFrame，成績資料表，需包含 '成績' 欄位
        current_context_info: dict，當前篩選條件資訊
        get_gemini_advice: function，呼叫 Gemini API 產生建議的函式
    回傳：
        無（直接於 Streamlit 畫面顯示）
    """
    st.subheader("成績分布與趨勢")
    if '成績' in df.columns:
        # 計算描述統計與分位數
        desc = df['成績'].describe().to_dict()
        quantiles = df['成績'].quantile([0.25, 0.5, 0.75]).to_dict()
        stats = {**desc, '分位數': quantiles}
        st.write(df['成績'].describe())
        # 靜態直方圖
        st.write("#### 成績分布直方圖 (靜態)")
        fig, ax = plt.subplots()
        sns.histplot(df['成績'], kde=True, ax=ax)
        st.pyplot(fig)
        # 互動直方圖（hover 顯示分數區間與人數）
        st.write("#### 成績分布直方圖 (互動式)")
        fig_px = px.histogram(df, x='成績', nbins=20, title='成績分布直方圖', marginal="box", histnorm=None)
        fig_px.update_traces(
            hovertemplate='分數區間: %{x}<br>人數: %{y}<extra></extra>'
        )
        st.plotly_chart(fig_px, use_container_width=True)
        # 呼叫 Gemini API 產生分析建議
        with st.spinner('正在產生AI分析建議，請稍候...'):
            st.info(get_gemini_advice('成績分布與趨勢', stats=stats, context_info=current_context_info))
    else:
        st.warning("本工作表無 '成績' 欄位") 