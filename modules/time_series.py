import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def time_series_tab(df, current_context_info, get_gemini_advice):
    st.subheader("時間序列分析（多檢測/分組趨勢）")
    st.markdown("""
    - 可觀察多次檢測、不同分組（如年級/學校）成績變化趨勢
    - 可切換顯示平均、標準差、及格率等指標
    """)
    if '日期' in df.columns and '成績' in df.columns:
        df.loc[:, '日期'] = pd.to_datetime(df['日期'], errors='coerce')
        test_options = df['檢測名稱'].dropna().unique().tolist() if '檢測名稱' in df.columns else []
        selected_tests = st.multiselect('選擇檢測名稱（可複選）', test_options, default=test_options)
        group_cols = [col for col in ['年級(匯出設定期末)','學校'] if col in df.columns]
        group_col = st.selectbox('分組欄位（可選）', ['無']+group_cols)
        metric = st.selectbox('趨勢指標', ['平均分數','標準差','及格率'])
        plot_df = df[df['檢測名稱'].isin(selected_tests)] if selected_tests else df
        if group_col != '無':
            groupby_cols = ['日期', group_col]
        else:
            groupby_cols = ['日期']
        if metric == '及格率':
            plot = plot_df.groupby(groupby_cols)['成績'].apply(lambda x: (x>=60).mean()*100).reset_index(name='及格率')
            trend = plot.pivot(index='日期', columns=group_col if group_col != '無' else None, values='及格率').to_dict() if group_col != '無' else plot.set_index('日期')['及格率'].to_dict()
        else:
            plot = plot_df.groupby(groupby_cols)['成績'].agg(['mean','std']).reset_index()
            trend = plot.pivot(index='日期', columns=group_col if group_col != '無' else None, values=metric=='平均分數' and 'mean' or 'std').to_dict() if group_col != '無' else plot.set_index('日期')[metric=='平均分數' and 'mean' or 'std'].to_dict()
        fig, ax = plt.subplots(figsize=(8,4))
        if group_col != '無':
            for key, grp in plot.groupby(group_col):
                if metric == '平均分數':
                    ax.plot(grp['日期'], grp['mean'], marker='o', label=str(key))
                elif metric == '標準差':
                    ax.plot(grp['日期'], grp['std'], marker='o', label=str(key))
                elif metric == '及格率':
                    ax.plot(grp['日期'], grp['及格率'], marker='o', label=str(key))
            ax.legend(title=group_col)
        else:
            if metric == '平均分數':
                ax.plot(plot['日期'], plot['mean'], marker='o')
            elif metric == '標準差':
                ax.plot(plot['日期'], plot['std'], marker='o')
            elif metric == '及格率':
                ax.plot(plot['日期'], plot['及格率'], marker='o')
        ax.set_xlabel('日期')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} 趨勢')
        plt.xticks(rotation=30)
        st.pyplot(fig)
        with st.spinner('正在產生AI分析建議，請稍候...'):
            st.info(get_gemini_advice('時間序列分析', stats={'metric': metric, '趨勢': trend}, context_info=current_context_info))
        st.info('指標說明：\n平均分數=該日/群組平均，標準差=分數離散程度，及格率=60分以上比例')
    else:
        st.warning("本工作表無 '日期' 或 '成績' 欄位") 