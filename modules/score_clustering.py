import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np

def score_clustering_tab(df, current_context_info, get_gemini_advice):
    """
    成績分群分析分頁：使用 KMeans 對學生成績自動分群，顯示分群統計、分布圖（含互動式），並呼叫 Gemini API 產生分群建議。
    
    參數：
        df: pd.DataFrame，成績資料表，需包含 '成績' 欄位
        current_context_info: dict，當前篩選條件資訊
        get_gemini_advice: function，呼叫 Gemini API 產生建議的函式
    回傳：
        無（直接於 Streamlit 畫面顯示）
    """
    st.subheader("成績分群分析（KMeans）")
    st.markdown("""
    - 依據成績自動分群，觀察不同分群的分布特性
    - 可用於辨識高分/中分/低分群學生
    """)
    if '成績' in df.columns:
        # 取得有效成績資料
        valid_scores = df['成績'].dropna().values.reshape(-1, 1)
        sse = []
        K_range = range(2, min(9, len(valid_scores)))
        # Elbow 法則計算不同分群數的 SSE
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(valid_scores)
            sse.append(kmeans.inertia_)
        # 畫出 Elbow 曲線
        fig_elbow, ax_elbow = plt.subplots()
        ax_elbow.plot(list(K_range), sse, marker='o')
        ax_elbow.set_xlabel('分群數量')
        ax_elbow.set_ylabel('SSE')
        ax_elbow.set_title('Elbow法則建議分群數')
        st.pyplot(fig_elbow)
        st.info('建議分群數量可參考Elbow圖拐點')
        # 讓使用者選擇分群數
        n_clusters = st.slider("選擇分群數量", min_value=2, max_value=8, value=3)
        if len(valid_scores) >= n_clusters:
            # 執行 KMeans 分群
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(valid_scores)
            df_clustered = df.copy()
            df_clustered.loc[df['成績'].notna(), '分群'] = clusters
            # 分群摘要統計
            summary = df_clustered.groupby('分群')['成績'].agg(['count','mean','std',lambda x: (x>=60).mean()*100])
            summary = summary.rename(columns={'<lambda_0>':'及格率(%)'})
            st.write('分群摘要統計：')
            st.dataframe(summary)
            # 顯示分群結果
            st.write(df_clustered[['姓名', '成績', '分群']].sort_values('分群'))
            # 下載分群結果
            csv = df_clustered[['姓名','成績','分群']].to_csv(index=False).encode('utf-8-sig')
            st.download_button('下載分群結果CSV', csv, file_name='分群結果.csv', mime='text/csv')
            # 靜態分群分布直方圖
            st.write("#### 分群分布直方圖 (靜態)")
            fig, ax = plt.subplots()
            sns.histplot(data=df_clustered, x='成績', hue='分群', multiple='stack', palette='tab10', ax=ax)
            st.pyplot(fig)
            # 互動分群分布直方圖（hover 顯示分群、分數、人數）
            st.write("#### 分群分布直方圖 (互動式)")
            fig_px = px.histogram(df_clustered, x='成績', color='分群', nbins=20, barmode='overlay', title='分群分布直方圖')
            fig_px.update_traces(
                hovertemplate='分群: %{customdata[0]}<br>分數: %{x}<br>人數: %{y}<extra></extra>',
                customdata=df_clustered[['分群']].values
            )
            st.plotly_chart(fig_px, use_container_width=True)
            # 彙整分群建議資訊
            group_advice = {}
            for k, v in summary.iterrows():
                group_advice[k] = {'mean': v['mean'], 'pass_rate': v['及格率(%)']}
            # 呼叫 Gemini API 產生分群建議
            st.info(get_gemini_advice('分群分析', group=group_advice, context_info=current_context_info))
        else:
            st.warning("有效成績數量不足以分成所選群數")
    else:
        st.warning("本工作表無 '成績' 欄位") 