import streamlit as st

def participation_tab(df, current_context_info, get_gemini_advice):
    st.subheader("檢測參與度")
    if '姓名' in df.columns:
        participation = df['姓名'].value_counts()
        stats = {
            '參與次數分布': participation.to_dict(),
            '平均參與次數': float(participation.mean()),
            '中位數參與次數': float(participation.median()),
            '最大參與次數': int(participation.max()),
            '最小參與次數': int(participation.min())
        }
        st.write(participation)
        with st.spinner('正在產生AI分析建議，請稍候...'):
            st.info(get_gemini_advice('檢測參與度', stats=stats, context_info=current_context_info))
    else:
        st.warning("本工作表無 '姓名' 欄位") 