import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import platform
import os

# 1. 한글 폰트 완벽 고정 설정
def setup_korean_font():
    plt.rcParams['axes.unicode_minus'] = False
    
    # OS별 한글 폰트 지정
    if platform.system() == 'Windows':
        font_name = 'Malgun Gothic'
        win_font_path = "C:/Windows/Fonts/malgun.ttf"
        if os.path.exists(win_font_path):
            fm.fontManager.addfont(win_font_path)
    elif platform.system() == 'Darwin':
        font_name = 'AppleGothic'
    else:
        font_name = 'NanumGothic'

    # Matplotlib 및 Seaborn 기본 폰트 강제 고정
    plt.rc('font', family=font_name)
    plt.rcParams['font.family'] = font_name
    plt.rcParams['font.sans-serif'] = [font_name, 'Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Gulim']

setup_korean_font()

# 2. 페이지 설정
st.set_page_config(page_title="무역 분석 대시보드", layout="wide")

# 3. 데이터 로드 및 상대국(j) 기준 매핑
@st.cache_data
def load_data():
    baci_df = pd.read_csv('baci_85_sample.csv')
    country_df = pd.read_csv('country_codes_sample.csv')
    
    baci_df.columns = baci_df.columns.str.strip()
    country_df.columns = country_df.columns.str.strip()
    
    code_col = country_df.columns[0]
    name_col = country_df.columns[1] if len(country_df.columns) > 1 else country_df.columns[0]
    
    for col in country_df.columns:
        if col.lower() in ['country_code', 'code', 'id', 'i', 'j', 'country_code_numeric']:
            code_col = col
        if col.lower() in ['country_name', 'name', 'country', 'country_name_full', 'country_name_abbreviation']:
            name_col = col

    baci_df['target_code'] = baci_df['j'].astype(str)
    country_df['clean_code'] = country_df[code_col].astype(str)
    
    merged_df = pd.merge(
        baci_df, 
        country_df[['clean_code', name_col]], 
        left_on='target_code', 
        right_on='clean_code', 
        how='left'
    )
    
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

# 5. 메인 화면 출력
st.title("무역 분석 대시보드")

# 결측치 확인
st.subheader("baci_85_sample.csv 파일의 결측치")
missing_df = baci_raw.isnull().sum().reset_index()
missing_df.columns = ['컬럼명', '결측치 수']
st.dataframe(missing_df, use_container_width=True)

# 지표 카드
col1, col2 = st.columns(2)
col1.metric("총거래건수", f"{len(filtered_df):,} 건")
col2.metric("총 수출액(달러)", f"${filtered_df['v'].sum():,.2f}")

st.markdown("---")

# 히트맵 & 바차트
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("국가*연도 수출액 히트맵(상위 8개국)")
    if not filtered_df.empty:
        setup_korean_font()
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
        setup_korean_font()
        grade_dist = filtered_df['무역액등급'].value_counts().reindex(['대', '중', '소'])
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        sns.barplot(x=grade_dist.index, y=grade_dist.values, palette="Blues_r", ax=ax2)
        ax2.set_xlabel("무역액 등급")
        ax2.set_ylabel("건수")
        st.pyplot(fig2)
    else:
        st.write("선택된 데이터가 없습니다.")

st.markdown("---")

# 상위 5개국 교차표
st.subheader("상위 5개국 * 무역액 등급 교차표")

if not filtered_df.empty:
    top_5_countries = filtered_df.groupby('country_name')['v'].sum().nlargest(5).index
    cross_target = filtered_df[filtered_df['country_name'].isin(top_5_countries)]
    
    col_raw, col_norm = st.columns(2)
    
    with col_raw:
        st.write("원본건수")
        raw_table = pd.crosstab(cross_target['country_name'], cross_target['무역액등급'])
        st.dataframe(raw_table, use_container_width=True)
        
    with col_norm:
        st.write("정규화비율")
        norm_table = pd.crosstab(cross_target['country_name'], cross_target['무역액등급'], normalize='index')
        st.dataframe(norm_table.style.format("{:.2%}"), use_container_width=True)
else:
    st.write("선택된 데이터가 없습니다.")