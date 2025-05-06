import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

def sidebar_filters(df):
    """
    建立 sidebar 篩選區，回傳篩選後的 df 及篩選資訊 dict
    """
    filter_cols = []
    st.sidebar.header("資料篩選區")
    # 學校篩選
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
    # 年級篩選
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
    # 檢測名稱分群
    auto_cluster_tests = []
    with st.sidebar.expander("檢測名稱分群", expanded=False):
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
    # 檢測名稱篩選
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
    return df, current_context_info 