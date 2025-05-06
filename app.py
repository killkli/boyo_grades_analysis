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

    # ===== 分頁式分析（Tabs）與首頁 Dashboard =====
    tab_names = ["Dashboard", "分數分布分析", "通過率分析", "成績分布與趨勢", "進階檢測與晉級分析", "檢測參與度", "時間序列分析", "個案追蹤", "交叉分析", "成績分群分析", "自動分群檢測分布分析"]
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
            st.write("#### 分數分布直方圖")
            fig1, ax1 = plt.subplots()
            sns.histplot(df['成績'], kde=True, ax=ax1)
            st.pyplot(fig1)
            st.write("#### 分數分布箱型圖")
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=df['成績'], ax=ax2)
            st.pyplot(fig2)
            st.info("指標說明：\n- 平均分數：所有學生分數的平均值。\n- 標準差：分數的離散程度，越大表示分數分布越分散。\n- 及格率：分數大於等於60分的學生比例。\n\n解讀建議：\n- 觀察平均分數與及格率可判斷整體學習狀況。\n- 標準差高時，代表學生間學習落差大。\n- 直方圖與箱型圖可協助辨識分數分布型態與極端值。")
        else:
            st.warning("本工作表無 '成績' 欄位")

    # 其餘分析分頁
    analysis_map = {
        2: "通過率分析",
        3: "成績分布與趨勢",
        4: "進階檢測與晉級分析",
        5: "檢測參與度",
        6: "時間序列分析",
        7: "個案追蹤",
        8: "交叉分析",
        9: "成績分群分析",
        10: "自動分群檢測分布分析"
    }
    for idx in range(2, len(tab_names)):
        with tabs[idx]:
            analysis_type = analysis_map[idx]
            if analysis_type == "通過率分析":
                st.subheader("通過率分析")
                if '是否通過？' in df.columns:
                    pass_rate = df['是否通過？'].value_counts(normalize=True) * 100
                    st.write(pass_rate)
                    fig, ax = plt.subplots()
                    pass_rate.plot(kind='bar', ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning("本工作表無 '是否通過？' 欄位")
            elif analysis_type == "成績分布與趨勢":
                st.subheader("成績分布與趨勢")
                if '成績' in df.columns:
                    st.write(df['成績'].describe())
                    fig, ax = plt.subplots()
                    sns.histplot(df['成績'], kde=True, ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning("本工作表無 '成績' 欄位")
            elif analysis_type == "進階檢測與晉級分析":
                st.subheader("進階檢測與晉級分析")
                if '是否計算為進階檢測？' in df.columns:
                    st.write(df['是否計算為進階檢測？'].value_counts())
                else:
                    st.warning("本工作表無 '是否計算為進階檢測？' 欄位")
            elif analysis_type == "檢測參與度":
                st.subheader("檢測參與度")
                if '姓名' in df.columns:
                    st.write(df['姓名'].value_counts())
                else:
                    st.warning("本工作表無 '姓名' 欄位")
            elif analysis_type == "時間序列分析":
                st.subheader("時間序列分析（多檢測/分組趨勢）")
                st.markdown("""
                - 可觀察多次檢測、不同分組（如年級/學校）成績變化趨勢
                - 可切換顯示平均、標準差、及格率等指標
                """)
                if '日期' in df.columns and '成績' in df.columns:
                    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
                    # 檢測名稱選擇
                    test_options = df['檢測名稱'].dropna().unique().tolist() if '檢測名稱' in df.columns else []
                    selected_tests = st.multiselect('選擇檢測名稱（可複選）', test_options, default=test_options)
                    # 分組欄位選擇
                    group_cols = [col for col in ['年級(匯出設定期末)','學校'] if col in df.columns]
                    group_col = st.selectbox('分組欄位（可選）', ['無']+group_cols)
                    # 指標選擇
                    metric = st.selectbox('趨勢指標', ['平均分數','標準差','及格率'])
                    plot_df = df[df['檢測名稱'].isin(selected_tests)] if selected_tests else df
                    if group_col != '無':
                        groupby_cols = ['日期', group_col]
                    else:
                        groupby_cols = ['日期']
                    agg_dict = {'平均分數':('成績','mean'), '標準差':('成績','std'), '及格率':(lambda x: (x>=60).mean()*100)}
                    if metric == '及格率':
                        plot = plot_df.groupby(groupby_cols)['成績'].apply(lambda x: (x>=60).mean()*100).reset_index(name='及格率')
                    else:
                        plot = plot_df.groupby(groupby_cols)['成績'].agg(['mean','std']).reset_index()
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
                    st.info('指標說明：\n平均分數=該日/群組平均，標準差=分數離散程度，及格率=60分以上比例')
                else:
                    st.warning("本工作表無 '日期' 或 '成績' 欄位")
            elif analysis_type == "個案追蹤":
                st.subheader("個案追蹤")
                if '姓名' in df.columns and '成績' in df.columns:
                    student = st.selectbox("選擇學生", df['姓名'].unique())
                    student_df = df[df['姓名'] == student]
                    st.write(student_df[['日期', '檢測名稱', '成績', '是否通過？']])
                    if '日期' in student_df.columns:
                        student_df['日期'] = pd.to_datetime(student_df['日期'], errors='coerce')
                        st.line_chart(student_df.sort_values('日期').set_index('日期')['成績'])
                else:
                    st.warning("本工作表無 '姓名' 或 '成績' 欄位")
            elif analysis_type == "交叉分析":
                st.subheader("交叉分析")
                if '是否通過？' in df.columns and '學生考核類別' in df.columns:
                    cross = pd.crosstab(df['學生考核類別'], df['是否通過？'])
                    st.write(cross)
                    st.bar_chart(cross)
                else:
                    st.warning("本工作表無 '是否通過？' 或 '學生考核類別' 欄位")
            elif analysis_type == "成績分群分析":
                st.subheader("成績分群分析（KMeans）")
                st.markdown("""
                - 依據成績自動分群，觀察不同分群的分布特性
                - 可用於辨識高分/中分/低分群學生
                """)
                if '成績' in df.columns:
                    # Elbow法則建議分群數
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
                        # 分群摘要
                        summary = df_clustered.groupby('分群')['成績'].agg(['count','mean','std',lambda x: (x>=60).mean()*100])
                        summary = summary.rename(columns={'<lambda_0>':'及格率(%)'})
                        st.write('分群摘要統計：')
                        st.dataframe(summary)
                        # 分群結果表
                        st.write(df_clustered[['姓名', '成績', '分群']].sort_values('分群'))
                        # 下載分群結果
                        csv = df_clustered[['姓名','成績','分群']].to_csv(index=False).encode('utf-8-sig')
                        st.download_button('下載分群結果CSV', csv, file_name='分群結果.csv', mime='text/csv')
                        # 分群分布圖
                        fig, ax = plt.subplots()
                        sns.histplot(data=df_clustered, x='成績', hue='分群', multiple='stack', palette='tab10', ax=ax)
                        st.pyplot(fig)
                        st.info('指標說明：\n分群依據KMeans演算法，分群數量可依Elbow法則建議調整。\n分群摘要顯示各群人數、平均、標準差、及格率。\n可下載分群結果進行後續追蹤。')
                    else:
                        st.warning("有效成績數量不足以分成所選群數")
                else:
                    st.warning("本工作表無 '成績' 欄位")
            elif analysis_type == "自動分群檢測分布分析":
                st.subheader("自動分群檢測分布分析")
                st.markdown("""
                - 依據檢測成績特徵（平均、標準差、樣本數）自動分群
                - 可觀察同類型檢測的分數分布差異
                """)
                if '檢測名稱' in df.columns and '成績' in df.columns:
                    # 先計算每個檢測名稱的分數分布特徵
                    test_features = df.groupby('檢測名稱')['成績'].agg(['mean', 'std', 'count']).fillna(0)
                    # Elbow法則建議分群數
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
                        test_features['分群'] = clusters
                        # 讓使用者選擇分群
                        selected_cluster = st.selectbox("選擇檢測分群", sorted(test_features['分群'].unique()))
                        selected_tests = test_features[test_features['分群'] == selected_cluster].index.tolist()
                        st.write(f"此分群包含檢測：{selected_tests}")
                        # 分群摘要
                        summary = test_features.groupby('分群').agg({'mean':'mean','std':'mean','count':'sum'})
                        summary = summary.rename(columns={'mean':'平均分數','std':'標準差','count':'檢測數'})
                        st.write('分群摘要統計：')
                        st.dataframe(summary)
                        # 分群結果下載
                        csv = test_features.reset_index()[['檢測名稱','mean','std','count','分群']].to_csv(index=False).encode('utf-8-sig')
                        st.download_button('下載分群檢測結果CSV', csv, file_name='檢測分群結果.csv', mime='text/csv')
                        # 顯示這群所有檢測的分數分布
                        filtered = df[df['檢測名稱'].isin(selected_tests)]
                        st.write(filtered[['檢測名稱', '成績']].groupby('檢測名稱').describe())
                        fig, ax = plt.subplots(figsize=(8,4))
                        sns.boxplot(data=filtered, x='檢測名稱', y='成績', ax=ax)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        st.info('指標說明：\n分群依據KMeans演算法，分群數量可依Elbow法則建議調整。\n分群摘要顯示各群平均分數、標準差、檢測數。\n可下載分群結果進行後續分析。')
                    else:
                        st.warning("檢測數量不足以分群")
                else:
                    st.warning("本工作表無 '檢測名稱' 或 '成績' 欄位")
else:
    st.info("請先上傳Excel檔案") 