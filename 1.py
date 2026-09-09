import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import platform
import os

# 1. 한글 폰트 설정
def setup_korean_font():
    local_fonts = [f for f in os.listdir('.') if f.endswith(('.ttf', '.otf'))]
    applied = False
    for font_path in local_fonts:
        try:
            prop = fm.FontProperties(fname=font_path)
            if "roboto" not in font_path.lower():
                font_name = prop.get_name()
                plt.rc('font', family=font_name)
                applied = True
                break
        except Exception:
            continue
            
    if not applied:
        system_name = platform.system()
        if system_name == 'Windows':
            plt.rc('font', family='Malgun Gothic')
        elif system_name == 'Darwin':
            plt.rc('font', family='AppleGothic')
        else:
            plt.rc('font', family='NanumGothic')

    plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()

# 2. 페이지 설정
st.set_page_config(page_title="무역 분석 대시보드", layout="wide")

# 3. 데이터 로드 및 교역 파트너국(j) 기준 매핑
@st.cache_data
def load_data():
    baci_df = pd.read_csv('baci_85_sample.csv')
    country_df = pd.read_csv('country_codes_sample.csv')
    
    baci_df.columns = baci_df.columns.str.strip()
    country_df.columns = country_df.columns.str.strip()
    
    # country_codes 파일의 코드 및 국가명 컬럼 자동 감지
    code_col = country_df.columns[0]
    name_col = country_df.columns[1] if len(country_df.columns) > 1 else country_df.columns[0]
    
    for col in country_df.columns:
        if col.lower() in ['country_code', 'code', 'id', 'i', 'j', 'country_code_numeric']:
            code_col = col
        if col.lower() in ['country_name', 'name', 'country', 'country_name_full', 'country_name_abbreviation']:
            name_col = col

    # 수입국/상대국(j) 기준 매핑 (수출국 i가 410 대한민국 단일값인 경우 대응)
    baci_df['target_code'] = baci_df['j'].astype(str)
    country_df['clean_code'] = country_df[code_col].astype(str)
    
    merged_df = pd.merge(
        baci_df, 
        country_df[['clean_code', name_col]], 
        left_on='target_code', 
        right_on='clean_code', 
        how='left'
    )
    
    # 국가명 부여
    merged_df['country_name'] = merged_df[name_col].fillna(merged_df['target_code'])
    
    # 무역액 등급 (대, 중, 소) 분할
    merged_df['무역액등급'] = pd.qcut(
        merged_df['v'], 
        q=3, 
        labels=['소', '중', '대']
    )
    
    return baci_df, merged_df

baci_raw, df = load_data()

# 4. 사이드바 필터
st.sidebar.header("필터 설정")

all_countries = sorted(df['country_name'].dropna().unique().tolist())

selected_countries = st.sidebar.multiselect(
    "국가 선택", 
    options=all_countries, 
    default=all_countries
)

grade_options = ['대', '중', '소']
selected_grades = st.sidebar.multiselect(
    "무역액 등급 선택", 
    options=grade_options, 
    default=grade_options
)

filtered_df = df[
    (df['country_name'].isin(selected_countries)) & 
    (df['무역액등급'].isin(selected_grades))
]

# 5. 메인 화면 구성
# 1. 타이틀
st.title("무역 분석 대시보드")

# 2. baci_85_sample.csv 파일의 결측치
st.subheader("baci_85_sample.csv 파일의 결측치")
missing_df = baci_raw.isnull().sum().reset_index()
missing_df.columns = ['컬럼명', '결측치 수']
st.dataframe(missing_df, use_container_width=True)

# 3. 총거래건수, 총 수출액(달러)
col1, col2 = st.columns(2)
col1.metric("총거래건수", f"{len(filtered_df):,} 건")
col2.metric("총 수출액(달러)", f"${filtered_df['v'].sum():,.2f}")

st.markdown("---")

# 4. 국가*연도 수출액 히트맵(상위 8개국) & 무역액 등급분포
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("국가*연도 수출액 히트맵(상위 8개국)")
    if not filtered_df.empty:
        top_8_countries = filtered_df.groupby('country_name')['v'].sum().nlargest(8).index
        heatmap_filtered = filtered_df[filtered_df['country_name'].isin(top_8_countries)]
        pivot_data = heatmap_filtered.pivot_table(index='country_name', columns='t', values='v', aggfunc='sum', fill_value=0)
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.heatmap(pivot_data, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
        ax.set_xlabel("연도")
        ax.set_ylabel("국가명")
        st.pyplot(fig)
    else:
        st.write("선택된 데이터가 없습니다.")

with col_right:
    st.subheader("무역액 등급분포")
    if not filtered_df.empty:
        grade_dist = filtered_df['무역액등급'].value_counts().reindex(['대', '중', '소'])
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        sns.barplot(x=grade_dist.index, y=grade_dist.values, palette="Blues_r", ax=ax2)
        ax2.set_xlabel("무역액 등급")
        ax2.set_ylabel("건수")
        st.pyplot(fig2)
    else:
        st.write("선택된 데이터가 없습니다.")

st.markdown("---")

# 5. 상위 5개국 * 무역액 등급 교차표 (원본건수 / 정규화비율)
st.subheader("상위 5개국 * 무역액 등급 교차표")

if not filtered_df.empty:
    top_5_countries = filtered_df.groupby('country_name')['v'].sum().nlargest(5).index
    cross_target = filtered_df[filtered_df['country_name'].isin(top_5_countries)]
    
    col_raw, col_norm = st.columns(2)
    
    with col_raw:
        st.markdown("**원본건수**")
        raw_table = pd.crosstab(cross_target['country_name'], cross_target['무역액등급'])
        st.dataframe(raw_table, use_container_width=True)
        
    with col_norm:
        st.markdown("**정규화비율**")
        norm_table = pd.crosstab(cross_target['country_name'], cross_target['무역액등급'], normalize='index')
        st.dataframe(norm_table.style.format("{:.2%}"), use_container_width=True)
else:
    st.write("선택된 데이터가 없습니다.")