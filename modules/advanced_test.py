import streamlit as st

def advanced_test_tab(df, current_context_info, get_gemini_advice):
    st.subheader("進階檢測與晉級分析")
    if '是否計算為進階檢測？' in df.columns:
        adv_stats = df['是否計算為進階檢測？'].value_counts().to_dict()
        st.write(df['是否計算為進階檢測？'].value_counts())
        with st.spinner('正在產生AI分析建議，請稍候...'):
            st.info(get_gemini_advice('進階檢測與晉級分析', stats=adv_stats, context_info=current_context_info))
    else:
        st.warning("本工作表無 '是否計算為進階檢測？' 欄位") 