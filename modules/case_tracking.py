import streamlit as st
import pandas as pd

def case_tracking_tab(df, current_context_info, get_gemini_advice):
    st.subheader("個案追蹤")
    if '姓名' in df.columns and '成績' in df.columns:
        student = st.selectbox("選擇學生", df['姓名'].unique())
        student_df = df[df['姓名'] == student]
        st.write(student_df[['日期', '檢測名稱', '成績', '是否通過？']])
        if '日期' in student_df.columns:
            student_df.loc[:, '日期'] = pd.to_datetime(student_df['日期'], errors='coerce')
            st.line_chart(student_df.sort_values('日期').set_index('日期')['成績'])
        case_stats = {
            '成績序列': student_df['成績'].tolist(),
            '檢測名稱序列': student_df['檢測名稱'].tolist() if '檢測名稱' in student_df.columns else [],
            '日期序列': student_df['日期'].dt.strftime('%Y-%m-%d').tolist() if '日期' in student_df.columns else [],
            '通過情形': student_df['是否通過？'].tolist() if '是否通過？' in student_df.columns else []
        }
        with st.spinner('正在產生AI分析建議，請稍候...'):
            st.info(get_gemini_advice('個案追蹤', stats=case_stats, group=[{'姓名': student}], context_info=current_context_info))
    else:
        st.warning("本工作表無 '姓名' 或 '成績' 欄位") 