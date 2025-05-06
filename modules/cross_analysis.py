import streamlit as st
import pandas as pd

def cross_analysis_tab(df, current_context_info, get_gemini_advice):
    st.subheader("交叉分析")
    if '是否通過？' in df.columns and '學生考核類別' in df.columns:
        cross = pd.crosstab(df['學生考核類別'], df['是否通過？'])
        st.write(cross)
        st.bar_chart(cross)
        with st.spinner('正在產生AI分析建議，請稍候...'):
            st.info(get_gemini_advice('交叉分析', stats={'交叉表': cross.to_dict()}, context_info=current_context_info))
    else:
        st.warning("本工作表無 '是否通過？' 或 '學生考核類別' 欄位") 