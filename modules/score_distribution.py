import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def score_distribution_tab(df, current_context_info, get_gemini_advice):
    """
    分數分布分析分頁：顯示成績的統計指標、分布直方圖與箱型圖（含互動式），並呼叫 Gemini API 產生分析建議。
    
    參數：
        df: pd.DataFrame，成績資料表，需包含 '成績' 欄位
        current_context_info: dict，當前篩選條件資訊
        get_gemini_advice: function，呼叫 Gemini API 產生建議的函式
    回傳：
        無（直接於 Streamlit 畫面顯示）
    """
    st.header("分數分布分析")
    if '成績' in df.columns:
        # 計算基本統計指標
        avg = df['成績'].mean()
        std = df['成績'].std()
        pass_rate = (df['成績'] >= 60).mean() * 100
        st.metric("平均分數", f"{avg:.2f}")
        st.metric("標準差", f"{std:.2f}")
        st.metric("及格率", f"{pass_rate:.1f}%")
        # 靜態直方圖
        st.write("#### 分數分布直方圖 (靜態)")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['成績'], kde=True, ax=ax1)
        st.pyplot(fig1)
        # 互動直方圖（hover 顯示分數區間與人數）
        st.write("#### 分數分布直方圖 (互動式)")
        hist_data = df['成績']
        fig1_px = px.histogram(df, x='成績', nbins=20, title='分數分布直方圖', marginal="box", histnorm=None)
        fig1_px.update_traces(
            hovertemplate='分數區間: %{x}<br>人數: %{y}<extra></extra>'
        )
        st.plotly_chart(fig1_px, use_container_width=True)
        # 靜態箱型圖
        st.write("#### 分數分布箱型圖 (靜態)")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x=df['成績'], ax=ax2)
        st.pyplot(fig2)
        # 互動箱型圖（hover 顯示分數與學生數）
        st.write("#### 分數分布箱型圖 (互動式)")
        fig2_px = px.box(df, y='成績', title='分數分布箱型圖')
        fig2_px.update_traces(
            hovertemplate='分數: %{y}<br>學生數: %{points.length}<extra></extra>'
        )
        st.plotly_chart(fig2_px, use_container_width=True)
        # 呼叫 Gemini API 產生分析建議
        st.info(get_gemini_advice('分數分布分析', stats={'avg': avg, 'std': std, 'pass_rate': pass_rate}, context_info=current_context_info))
    else:
        st.warning("本工作表無 '成績' 欄位") 