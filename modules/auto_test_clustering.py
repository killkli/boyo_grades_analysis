import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans

def auto_test_clustering_tab(df, current_context_info, get_gemini_advice):
    """
    自動分群檢測分布分析分頁：依據檢測成績特徵自動分群，顯示分群摘要、檢測分布、互動圖表，並呼叫 Gemini API 產生建議。
    
    參數：
        df: pd.DataFrame，成績資料表，需包含 '檢測名稱'、'成績' 欄位
        current_context_info: dict，當前篩選條件資訊
        get_gemini_advice: function，呼叫 Gemini API 產生建議的函式
    回傳：
        無（直接於 Streamlit 畫面顯示）
    """
    st.subheader("自動分群檢測分布分析")
    st.markdown("""
    - 依據檢測成績特徵（平均、標準差、樣本數）自動分群
    - 可觀察同類型檢測的分數分布差異
    """)
    if '檢測名稱' in df.columns and '成績' in df.columns:
        # 計算每個檢測的特徵
        test_features = df.groupby('檢測名稱')['成績'].agg(['mean', 'std', 'count']).fillna(0)
        sse = []
        K_range = range(2, min(9, len(test_features)))
        # Elbow 法則計算不同分群數的 SSE
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(test_features[['mean', 'std', 'count']])
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
        n_clusters = st.slider("分群數量", min_value=2, max_value=min(8, len(test_features)), value=3)
        if len(test_features) >= n_clusters:
            # 執行 KMeans 分群
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(test_features[['mean', 'std', 'count']])
            test_features.loc[:, '分群'] = clusters
            # 選擇分群顯示檢測
            selected_cluster = st.selectbox("選擇檢測分群", sorted(test_features['分群'].unique()))
            selected_tests = test_features[test_features['分群'] == selected_cluster].index.tolist()
            st.write(f"此分群包含檢測：{selected_tests}")
            # 分群摘要統計
            summary = test_features.groupby('分群').agg({'mean':'mean','std':'mean','count':'sum'})
            summary = summary.rename(columns={'mean':'平均分數','std':'標準差','count':'檢測數'})
            st.write('分群摘要統計：')
            st.dataframe(summary)
            # 下載分群檢測結果
            csv = test_features.reset_index()[['檢測名稱','mean','std','count','分群']].to_csv(index=False).encode('utf-8-sig')
            st.download_button('下載分群檢測結果CSV', csv, file_name='檢測分群結果.csv', mime='text/csv')
            # 顯示分群檢測分布描述
            filtered = df[df['檢測名稱'].isin(selected_tests)]
            st.write(filtered[['檢測名稱', '成績']].groupby('檢測名稱').describe())
            # 靜態箱型圖
            fig, ax = plt.subplots(figsize=(8,4))
            sns.boxplot(data=filtered, x='檢測名稱', y='成績', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            # 呼叫 Gemini API 產生分析建議
            with st.spinner('正在產生AI分析建議，請稍候...'):
                st.info(get_gemini_advice('自動分群檢測分布分析', stats=summary.to_dict(), context_info=current_context_info))
            st.info('指標說明：\n分群依據KMeans演算法，分群數量可依Elbow法則建議調整。\n分群摘要顯示各群平均分數、標準差、檢測數。\n可下載分群結果進行後續分析。')
            # 若需互動圖表，可於此加入 Plotly 互動箱型圖或直方圖
            # 例如：
            # import plotly.express as px
            # fig_px = px.box(filtered, x='檢測名稱', y='成績', color='檢測名稱', title='分群檢測分布箱型圖')
            # fig_px.update_traces(hovertemplate='檢測名稱: %{x}<br>分數: %{y}<extra></extra>')
            # st.plotly_chart(fig_px, use_container_width=True)
        else:
            st.warning("檢測數量不足以分群")
    else:
        st.warning("本工作表無 '檢測名稱' 或 '成績' 欄位") 