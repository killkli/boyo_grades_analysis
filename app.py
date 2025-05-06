import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from io import BytesIO
from sklearn.cluster import KMeans
import numpy as np
import matplotlib
import platform
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression
import random
import os
import google.generativeai as genai
import plotly.express as px


def get_gemini_advice(context, stats=None, group=None, context_info=None):
    """
    呼叫 Gemini API 產生分析建議，若無 API 金鑰則 fallback 為 mock。
    context: 分析區塊主題
    stats: 相關統計數據（dict）
    group: 分群標籤或學生名單
    context_info: dict, e.g. {'學校': [...], '年級': [...], '檢測名稱': [...]}
    """
    api_key = os.environ.get('GEMINI_API_KEY')
    prompt = ""
    info_str = ""
    if context_info:
        info_str = "分析對象資訊："
        for k, v in context_info.items():
            if v:
                info_str += f"{k}：{', '.join(map(str, v))}；"
        if info_str:
            info_str += "\n"
    if context == '分數分布分析':
        avg = stats.get('avg', 0)
        std = stats.get('std', 0)
        pass_rate = stats.get('pass_rate', 0)
        prompt = f"{info_str}請根據以下數據，產生一段專業且具教育意義的成績分布分析建議，並給出指標說明與解讀建議：\n平均分數：{avg:.2f}，標準差：{std:.2f}，及格率：{pass_rate:.1f}%。"
    elif context == '分群分析':
        if group is not None and isinstance(group, dict):
            group_str = '\n'.join(
                [f"群組 {k}: 平均 {v['mean']:.1f}，及格率 {v['pass_rate']:.1f}%" for k, v in group.items()])
            prompt = f"{info_str}請根據以下分群摘要，產生一段專業且具教育意義的分群分析建議，針對不同分群給予個人化學習建議：\n{group_str}"
        else:
            prompt = f"{info_str}請產生一段分群分析的教育建議。"
    elif context == '預測分析':
        if group is not None and isinstance(group, list):
            if not group:
                return "無高風險學生。"
            group_str = '\n'.join(
                [f"{stu['姓名']}：預測分數 {stu['預測分數']:.1f}" for stu in group])
            prompt = f"{info_str}以下為高風險學生名單，請針對每位學生給予個人化學習建議：\n{group_str}"
        else:
            prompt = f"{info_str}請產生一段預測分析的教育建議。"
    else:
        prompt = f"{info_str}請針對本區塊分析數據，產生一段教育意義建議。"
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name='gemini-2.0-flash')
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"[Gemini API 錯誤 fallback] {str(e)}\n{mock_gemini_advice(context, stats, group, context_info)}"
    else:
        return mock_gemini_advice(context, stats, group, context_info)


def mock_gemini_advice(context, stats=None, group=None, context_info=None):
    info_str = ""
    if context_info:
        info_str = "分析對象資訊："
        for k, v in context_info.items():
            if v:
                info_str += f"{k}：{', '.join(map(str, v))}；"
        if info_str:
            info_str += "\n"
    if context == '分數分布分析':
        avg = stats.get('avg', 0)
        std = stats.get('std', 0)
        pass_rate = stats.get('pass_rate', 0)
        advice = f"{info_str}平均分數為 {avg:.2f}，標準差 {std:.2f}，及格率 {pass_rate:.1f}%。\n"
        if avg >= 80:
            advice += "整體表現優異，可持續精進。"
        elif avg >= 60:
            advice += "大多數學生已達及格，建議針對落後學生加強輔導。"
        else:
            advice += "平均分數偏低，建議檢視教學內容與學習策略。"
        if std > 15:
            advice += "\n分數分布離散，學生間學習落差大。"
        return advice
    elif context == '分群分析':
        advice = info_str if info_str else ""
        if group is not None and isinstance(group, dict):
            advice += "分群摘要：\n"
            for k, v in group.items():
                advice += f"群組 {k}: 平均 {v['mean']:.1f}，及格率 {v['pass_rate']:.1f}%\n"
                if v['mean'] < 60:
                    advice += "→ 建議加強基礎教學。\n"
                elif v['mean'] > 80:
                    advice += "→ 可安排進階學習。\n"
            return advice
        return advice + "分群有助於辨識不同學習層次學生，建議針對低分群給予補救教學。"
    elif context == '預測分析':
        advice = info_str if info_str else ""
        if group is not None and isinstance(group, list):
            if not group:
                return advice + "無高風險學生。"
            advice += "高風險學生建議：\n"
            for stu in group:
                advice += f"{stu['姓名']}：預測分數 {stu['預測分數']:.1f}，建議加強輔導。\n"
            return advice
        return advice + "預測分析可協助及早發現學習風險，建議針對預測分數低於及格線學生進行輔導。"
    return info_str + "本區塊分析有助於掌握學生學習狀況，建議依據數據調整教學策略。"

def get_available_font(font_list):
    available = set(f.name for f in fm.fontManager.ttflist)
    for font in font_list:
        if font in available:
            return font
    return None

system = platform.system()
if system == 'Windows':
    font_candidates = ['Microsoft JhengHei', 'DFKai-SB', 'SimHei', 'SimSun']
elif system == 'Darwin':  # macOS
    font_candidates = ['PingFang TC', 'Heiti TC', '黑體-繁', 'STHeiti', 'Bpmf GenYo Gothic', 'Bpmf GenSeki Gothic', 'Bpmf Zihi Sans']
else:  # Linux 或其他
    font_candidates = ['Noto Sans TC', 'AR PL UMing TW', 'AR PL UKai TW', 'Source Han Serif TC', 'Source Han Sans TC']

font_name = get_available_font(font_candidates)
if font_name:
    matplotlib.rc('font', family=font_name)
else:
    matplotlib.rc('font', family='Heiti TC')  # fallback
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="成績分析App", layout="wide")
st.title("學生成績Excel分析App")

uploaded_file = st.file_uploader("請上傳成績Excel檔案", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    st.sidebar.header("資料篩選區")
    sheet = st.sidebar.selectbox("選擇工作表", sheet_names)
    df = pd.read_excel(xls, sheet_name=sheet)
    st.write(f"### 預覽：{sheet}")
    st.dataframe(df.head(20))

    # ===== Sidebar 篩選區分區明確，使用 expander 分組 =====
    filter_cols = []
    with st.sidebar.expander("學校篩選", expanded=True):
        if '學校' in df.columns:
            schools = df['學校'].dropna()
            school_counts = schools.value_counts()
            schools_sorted = school_counts.index.tolist()
            selected_school = st.multiselect('學校', schools_sorted, default=schools_sorted, help='可搜尋學校')
            col1, col2 = st.columns(2)
            with col1:
                if st.button('全選學校'):
                    selected_school = schools_sorted
            with col2:
                if st.button('全不選學校'):
                    selected_school = []
            filter_cols.append(('學校', selected_school))
    with st.sidebar.expander("年級篩選", expanded=False):
        if '年級(匯出設定期末)' in df.columns:
            grades = df['年級(匯出設定期末)'].dropna()
            grade_counts = grades.value_counts()
            grades_sorted = grade_counts.index.tolist()
            selected_grade = st.multiselect('年級', grades_sorted, default=grades_sorted, help='可搜尋年級')
            col1, col2 = st.columns(2)
            with col1:
                if st.button('全選年級'):
                    selected_grade = grades_sorted
            with col2:
                if st.button('全不選年級'):
                    selected_grade = []
            filter_cols.append(('年級(匯出設定期末)', selected_grade))
    with st.sidebar.expander("檢測名稱分群", expanded=False):
        auto_cluster_tests = []
        if '檢測名稱' in df.columns and '成績' in df.columns and len(df['檢測名稱'].unique()) > 1:
            test_features = df.groupby('檢測名稱')['成績'].agg(['mean', 'std', 'count']).fillna(0)
            n_clusters = st.slider("檢測名稱分群數量", min_value=2, max_value=min(8, len(test_features)), value=3, key='cluster_slider')
            if len(test_features) >= n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(test_features[['mean', 'std', 'count']])
                test_features['分群'] = clusters
                cluster_options = sorted(test_features['分群'].unique())
                selected_cluster = st.selectbox("依分群選擇檢測", cluster_options, key='cluster_select')
                auto_cluster_tests = test_features[test_features['分群'] == selected_cluster].index.tolist()
                st.write(f"此分群包含檢測：{auto_cluster_tests}")
                if st.button('套用此分群檢測到下方篩選'):
                    st.session_state['selected_test'] = auto_cluster_tests
    with st.sidebar.expander("檢測名稱篩選", expanded=False):
        if '檢測名稱' in df.columns:
            tests = df['檢測名稱'].dropna()
            test_counts = tests.value_counts()
            tests_sorted = test_counts.index.tolist()
            if 'selected_test' not in st.session_state:
                st.session_state['selected_test'] = tests_sorted
            selected_test = st.multiselect('檢測名稱', tests_sorted, default=st.session_state['selected_test'], help='可搜尋檢測名稱', key='test_multiselect')
            col1, col2 = st.columns(2)
            with col1:
                if st.button('全選檢測名稱'):
                    selected_test = tests_sorted
                    st.session_state['selected_test'] = tests_sorted
            with col2:
                if st.button('全不選檢測名稱'):
                    selected_test = []
                    st.session_state['selected_test'] = []
            if auto_cluster_tests and st.session_state.get('selected_test') != auto_cluster_tests:
                selected_test = auto_cluster_tests
                st.session_state['selected_test'] = auto_cluster_tests
            filter_cols.append(('檢測名稱', selected_test))
    # 篩選資料
    for col, selected in filter_cols:
        if selected:
            df = df[df[col].isin(selected)]

    # 取得目前篩選資訊
    current_context_info = {col: selected for col, selected in filter_cols}

    # ===== 分頁式分析（Tabs）與首頁 Dashboard =====
    tab_names = ["Dashboard", "分數分布分析", "通過率分析", "成績分布與趨勢", "進階檢測與晉級分析", "檢測參與度", "時間序列分析", "個案追蹤", "交叉分析", "成績分群分析", "自動分群檢測分布分析", "相關性分析", "預測分析"]
    tabs = st.tabs(tab_names)

    # Dashboard
    with tabs[0]:
        st.header("Dashboard 總覽")
        col1, col2, col3 = st.columns(3)
        if '姓名' in df.columns:
            col1.metric("學生人數", df['姓名'].nunique())
        if '成績' in df.columns:
            col2.metric("平均分數", f"{df['成績'].mean():.2f}")
            col3.metric("及格率", f"{(df['成績']>=60).mean()*100:.1f}%")
            fig, ax = plt.subplots()
            sns.histplot(df['成績'], kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.info("本資料無成績欄位")

    # 分數分布分析
    with tabs[1]:
        st.header("分數分布分析")
        if '成績' in df.columns:
            avg = df['成績'].mean()
            std = df['成績'].std()
            pass_rate = (df['成績'] >= 60).mean() * 100
            st.metric("平均分數", f"{avg:.2f}")
            st.metric("標準差", f"{std:.2f}")
            st.metric("及格率", f"{pass_rate:.1f}%")
            st.write("#### 分數分布直方圖 (靜態)")
            fig1, ax1 = plt.subplots()
            sns.histplot(df['成績'], kde=True, ax=ax1)
            st.pyplot(fig1)
            st.write("#### 分數分布直方圖 (互動式)")
            fig1_px = px.histogram(df, x='成績', nbins=20, title='分數分布直方圖', marginal="box", histnorm=None)
            st.plotly_chart(fig1_px, use_container_width=True)
            st.write("#### 分數分布箱型圖 (靜態)")
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=df['成績'], ax=ax2)
            st.pyplot(fig2)
            st.write("#### 分數分布箱型圖 (互動式)")
            fig2_px = px.box(df, y='成績', title='分數分布箱型圖')
            st.plotly_chart(fig2_px, use_container_width=True)
            st.info(get_gemini_advice('分數分布分析', stats={'avg': avg, 'std': std, 'pass_rate': pass_rate}, context_info=current_context_info))
        else:
            st.warning("本工作表無 '成績' 欄位")

    # 其餘分析分頁
    for idx, tab_name in enumerate(tab_names):
        # 跳過 Dashboard、分數分布分析、相關性分析、預測分析（這些已在上方 with 處理）
        if tab_name in ["Dashboard", "分數分布分析", "相關性分析", "預測分析"]:
            continue
        with tabs[idx]:
            if tab_name == "通過率分析":
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
            elif tab_name == "成績分布與趨勢":
                st.subheader("成績分布與趨勢")
                if '成績' in df.columns:
                    desc = df['成績'].describe().to_dict()
                    quantiles = df['成績'].quantile([0.25, 0.5, 0.75]).to_dict()
                    stats = {**desc, '分位數': quantiles}
                    st.write(df['成績'].describe())
                    st.write("#### 成績分布直方圖 (靜態)")
                    fig, ax = plt.subplots()
                    sns.histplot(df['成績'], kde=True, ax=ax)
                    st.pyplot(fig)
                    st.write("#### 成績分布直方圖 (互動式)")
                    fig_px = px.histogram(df, x='成績', nbins=20, title='成績分布直方圖', marginal="box", histnorm=None)
                    st.plotly_chart(fig_px, use_container_width=True)
                    with st.spinner('正在產生AI分析建議，請稍候...'):
                        st.info(get_gemini_advice('成績分布與趨勢', stats=stats, context_info=current_context_info))
                else:
                    st.warning("本工作表無 '成績' 欄位")
            elif tab_name == "進階檢測與晉級分析":
                st.subheader("進階檢測與晉級分析")
                if '是否計算為進階檢測？' in df.columns:
                    adv_stats = df['是否計算為進階檢測？'].value_counts().to_dict()
                    st.write(df['是否計算為進階檢測？'].value_counts())
                    with st.spinner('正在產生AI分析建議，請稍候...'):
                        st.info(get_gemini_advice('進階檢測與晉級分析', stats=adv_stats, context_info=current_context_info))
                else:
                    st.warning("本工作表無 '是否計算為進階檢測？' 欄位")
            elif tab_name == "檢測參與度":
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
            elif tab_name == "時間序列分析":
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
            elif tab_name == "個案追蹤":
                st.subheader("個案追蹤")
                if '姓名' in df.columns and '成績' in df.columns:
                    student = st.selectbox("選擇學生", df['姓名'].unique())
                    student_df = df[df['姓名'] == student]
                    st.write(student_df[['日期', '檢測名稱', '成績', '是否通過？']])
                    if '日期' in student_df.columns:
                        student_df.loc[:, '日期'] = pd.to_datetime(student_df['日期'], errors='coerce')
                        st.line_chart(student_df.sort_values('日期').set_index('日期')['成績'])
                    # 個案追蹤給該生所有成績序列、檢測名稱、通過情形
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
            elif tab_name == "交叉分析":
                st.subheader("交叉分析")
                if '是否通過？' in df.columns and '學生考核類別' in df.columns:
                    cross = pd.crosstab(df['學生考核類別'], df['是否通過？'])
                    st.write(cross)
                    st.bar_chart(cross)
                    with st.spinner('正在產生AI分析建議，請稍候...'):
                        st.info(get_gemini_advice('交叉分析', stats={'交叉表': cross.to_dict()}, context_info=current_context_info))
                else:
                    st.warning("本工作表無 '是否通過？' 或 '學生考核類別' 欄位")
            elif tab_name == "成績分群分析":
                st.subheader("成績分群分析（KMeans）")
                st.markdown("""
                - 依據成績自動分群，觀察不同分群的分布特性
                - 可用於辨識高分/中分/低分群學生
                """)
                if '成績' in df.columns:
                    valid_scores = df['成績'].dropna().values.reshape(-1, 1)
                    sse = []
                    K_range = range(2, min(9, len(valid_scores)))
                    for k in K_range:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(valid_scores)
                        sse.append(kmeans.inertia_)
                    fig_elbow, ax_elbow = plt.subplots()
                    ax_elbow.plot(list(K_range), sse, marker='o')
                    ax_elbow.set_xlabel('分群數量')
                    ax_elbow.set_ylabel('SSE')
                    ax_elbow.set_title('Elbow法則建議分群數')
                    st.pyplot(fig_elbow)
                    st.info('建議分群數量可參考Elbow圖拐點')
                    n_clusters = st.slider("選擇分群數量", min_value=2, max_value=8, value=3)
                    if len(valid_scores) >= n_clusters:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(valid_scores)
                        df_clustered = df.copy()
                        df_clustered.loc[df['成績'].notna(), '分群'] = clusters
                        summary = df_clustered.groupby('分群')['成績'].agg(['count','mean','std',lambda x: (x>=60).mean()*100])
                        summary = summary.rename(columns={'<lambda_0>':'及格率(%)'})
                        st.write('分群摘要統計：')
                        st.dataframe(summary)
                        st.write(df_clustered[['姓名', '成績', '分群']].sort_values('分群'))
                        csv = df_clustered[['姓名','成績','分群']].to_csv(index=False).encode('utf-8-sig')
                        st.download_button('下載分群結果CSV', csv, file_name='分群結果.csv', mime='text/csv')
                        st.write("#### 分群分布直方圖 (靜態)")
                        fig, ax = plt.subplots()
                        sns.histplot(data=df_clustered, x='成績', hue='分群', multiple='stack', palette='tab10', ax=ax)
                        st.pyplot(fig)
                        st.write("#### 分群分布直方圖 (互動式)")
                        fig_px = px.histogram(df_clustered, x='成績', color='分群', nbins=20, barmode='overlay', title='分群分布直方圖')
                        st.plotly_chart(fig_px, use_container_width=True)
                        # 個人化分群建議
                        group_advice = {}
                        for k, v in summary.iterrows():
                            group_advice[k] = {'mean': v['mean'], 'pass_rate': v['及格率(%)']}
                        st.info(get_gemini_advice('分群分析', group=group_advice, context_info=current_context_info))
                    else:
                        st.warning("有效成績數量不足以分成所選群數")
                else:
                    st.warning("本工作表無 '成績' 欄位")
            elif tab_name == "自動分群檢測分布分析":
                st.subheader("自動分群檢測分布分析")
                st.markdown("""
                - 依據檢測成績特徵（平均、標準差、樣本數）自動分群
                - 可觀察同類型檢測的分數分布差異
                """)
                if '檢測名稱' in df.columns and '成績' in df.columns:
                    test_features = df.groupby('檢測名稱')['成績'].agg(['mean', 'std', 'count']).fillna(0)
                    sse = []
                    K_range = range(2, min(9, len(test_features)))
                    for k in K_range:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(test_features[['mean', 'std', 'count']])
                        sse.append(kmeans.inertia_)
                    fig_elbow, ax_elbow = plt.subplots()
                    ax_elbow.plot(list(K_range), sse, marker='o')
                    ax_elbow.set_xlabel('分群數量')
                    ax_elbow.set_ylabel('SSE')
                    ax_elbow.set_title('Elbow法則建議分群數')
                    st.pyplot(fig_elbow)
                    st.info('建議分群數量可參考Elbow圖拐點')
                    n_clusters = st.slider("分群數量", min_value=2, max_value=min(8, len(test_features)), value=3)
                    if len(test_features) >= n_clusters:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(test_features[['mean', 'std', 'count']])
                        test_features.loc[:, '分群'] = clusters
                        selected_cluster = st.selectbox("選擇檢測分群", sorted(test_features['分群'].unique()))
                        selected_tests = test_features[test_features['分群'] == selected_cluster].index.tolist()
                        st.write(f"此分群包含檢測：{selected_tests}")
                        summary = test_features.groupby('分群').agg({'mean':'mean','std':'mean','count':'sum'})
                        summary = summary.rename(columns={'mean':'平均分數','std':'標準差','count':'檢測數'})
                        st.write('分群摘要統計：')
                        st.dataframe(summary)
                        csv = test_features.reset_index()[['檢測名稱','mean','std','count','分群']].to_csv(index=False).encode('utf-8-sig')
                        st.download_button('下載分群檢測結果CSV', csv, file_name='檢測分群結果.csv', mime='text/csv')
                        filtered = df[df['檢測名稱'].isin(selected_tests)]
                        st.write(filtered[['檢測名稱', '成績']].groupby('檢測名稱').describe())
                        fig, ax = plt.subplots(figsize=(8,4))
                        sns.boxplot(data=filtered, x='檢測名稱', y='成績', ax=ax)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        with st.spinner('正在產生AI分析建議，請稍候...'):
                            st.info(get_gemini_advice('自動分群檢測分布分析', stats=summary.to_dict(), context_info=current_context_info))
                        st.info('指標說明：\n分群依據KMeans演算法，分群數量可依Elbow法則建議調整。\n分群摘要顯示各群平均分數、標準差、檢測數。\n可下載分群結果進行後續分析。')
                    else:
                        st.warning("檢測數量不足以分群")
                else:
                    st.warning("本工作表無 '檢測名稱' 或 '成績' 欄位")

    # 相關性分析
    with tabs[-2]:
        st.header("相關性分析")
        # 依據資料欄位決定可分析單位
        options = []
        if '檢測名稱' in df.columns:
            options.append('檢測名稱')
        if '科目' in df.columns:
            options.append('科目')
        if not options:
            st.warning("本工作表無可用於相關性分析的欄位（需有 '檢測名稱' 或 '科目'）")
        else:
            unit = st.selectbox('選擇分析單位', options)
            # 轉成 wide 格式
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

    # 預測分析
    with tabs[-1]:
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
                # 個人化預測建議
                high_risk = pred_df[pred_df['風險預警']=='高風險'][['姓名','預測分數']].to_dict('records')
                st.info(get_gemini_advice('預測分析', group=high_risk, context_info=current_context_info))
            else:
                st.info('無足夠多次成績紀錄的學生可進行預測（需至少3次成績）')
        else:
            st.warning("本工作表需有 '姓名'、'成績'、'日期' 欄位")
else:
    st.info("請先上傳Excel檔案")

