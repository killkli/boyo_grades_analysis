import streamlit as st
import matplotlib.pyplot as plt

def pass_rate_tab(df, current_context_info, get_gemini_advice):
    st.subheader("通過率分析")
    if '是否通過？' in df.columns:
        pass_rate = df['是否通過？'].value_counts(normalize=True) * 100
        count = df['是否通過？'].value_counts()
        stats = {
            '通過率': pass_rate.to_dict(),
            '人數': count.to_dict()
        }
        st.write(pass_rate)
        fig, ax = plt.subplots()
        pass_rate.plot(kind='bar', ax=ax)
        st.pyplot(fig)
        with st.spinner('正在產生AI分析建議，請稍候...'):
            st.info(get_gemini_advice('通過率分析', stats=stats, context_info=current_context_info))
    else:
        st.warning("本工作表無 '是否通過？' 欄位") 