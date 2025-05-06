import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def correlation_tab(df):
    st.header("相關性分析")
    options = []
    if '檢測名稱' in df.columns:
        options.append('檢測名稱')
    if '科目' in df.columns:
        options.append('科目')
    if not options:
        st.warning("本工作表無可用於相關性分析的欄位（需有 '檢測名稱' 或 '科目'）")
    else:
        unit = st.selectbox('選擇分析單位', options)
        if unit == '檢測名稱':
            pivot = df.pivot_table(index='姓名', columns='檢測名稱', values='成績')
        else:
            pivot = df.pivot_table(index='姓名', columns='科目', values='成績')
        corr = pivot.corr(method='pearson')
        st.write('#### 相關係數矩陣')
        st.dataframe(corr.round(2))
        st.write('#### 相關係數熱力圖')
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)
        st.info("指標說明：\n- 皮爾森相關係數介於-1到1，越接近1表示正相關，越接近-1表示負相關，0表示無線性相關。\n\n解讀建議：\n- 相關係數高的檢測/科目，學生表現有高度一致性，可能有共同學習基礎。\n- 相關係數低或負相關，代表表現差異大，建議進一步分析原因。\n- 熱力圖可快速辨識高低相關的檢測/科目組合。") 