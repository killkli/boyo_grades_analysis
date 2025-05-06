import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def prediction_tab(df, current_context_info, get_gemini_advice):
    st.header("預測分析（學習風險預警/未來表現預測）")
    if '姓名' in df.columns and '成績' in df.columns and '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        pred_results = []
        for name, group in df.sort_values('日期').groupby('姓名'):
            if group['成績'].notna().sum() >= 3:
                group = group.dropna(subset=['成績', '日期'])
                group = group.sort_values('日期')
                X = np.arange(len(group)).reshape(-1,1)
                y = group['成績'].values
                model = LinearRegression()
                model.fit(X, y)
                next_idx = np.array([[len(group)]])
                pred_score = model.predict(next_idx)[0]
                pred_results.append({'姓名': name, '預測分數': pred_score, '最近分數': y[-1]})
        if pred_results:
            pred_df = pd.DataFrame(pred_results)
            pred_df['風險預警'] = np.where(pred_df['預測分數'] < 60, '高風險', '正常')
            st.write('#### 學生未來分數預測')
            st.dataframe(pred_df.round(2))
            st.write('#### 高風險學生名單（預測分數<60）')
            st.dataframe(pred_df[pred_df['風險預警']=='高風險'][['姓名','預測分數','最近分數']].round(2))
            st.write('#### 預測分數分布圖')
            fig, ax = plt.subplots()
            sns.histplot(pred_df['預測分數'], bins=20, kde=True, ax=ax)
            ax.axvline(60, color='red', linestyle='--', label='及格線')
            ax.legend()
            st.pyplot(fig)
            high_risk = pred_df[pred_df['風險預警']=='高風險'][['姓名','預測分數']].to_dict('records')
            st.info(get_gemini_advice('預測分析', group=high_risk, context_info=current_context_info))
        else:
            st.info('無足夠多次成績紀錄的學生可進行預測（需至少3次成績）')
    else:
        st.warning("本工作表需有 '姓名'、'成績'、'日期' 欄位") 