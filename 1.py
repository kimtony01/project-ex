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

# 3. 데이터 로드 및 국가명 매핑
ISO_NUMERIC_TO_NAME = {
    410: "대한민국 (Korea)",
    842: "미국 (USA)",
    156: "중국 (China)",
    392: "일본 (Japan)",
    276: "독일 (Germany)",
    704: "베트남 (Vietnam)",
    344: "홍콩 (Hong Kong)",
    158: "대만 (Taiwan)",
    702: "싱가포르 (Singapore)",
    458: "말레이시아 (Malaysia)",
    360: "인도네시아 (Indonesia)",
    764: "태국 (Thailand)",
    699: "인도 (India)",
    356: "인도 (India)",
    251: "프랑스 (France)",
    826: "영국 (UK)",
    381: "이탈리아 (Italy)",
    528: "네덜란드 (Netherlands)",
    757: "스위스 (Switzerland)",
    36: "호주 (Australia)",
    124: "캐나다 (Canada)",
    484: "멕시코 (Mexico)",
    76: "브라질 (Brazil)",
    643: "러시아 (Russia)"
}

@st.cache_data
def load_data():
    baci_df = pd.read_csv('baci_85_sample.csv')
    baci_df.columns = baci_df.columns.str.strip()
    
    country_map = {}
    if os.path.exists('country_codes_sample.csv'):
        country_df = pd.read_csv('country_codes_sample.csv')
        country_df.columns = country_df.columns.str.strip()
        
        code_col = country_df.columns[0]
        name_col = country_df.columns[1] if len(country_df.columns) > 1 else country_df.columns[0]
        
        for _, row in country_df.iterrows():
            try:
                c_code = int(float(row[code_col]))
                c_name = str(row[name_col]).strip()
                country_map[c_code] = c_name
            except Exception:
                continue

    for k, v in ISO_NUMERIC_TO_NAME.items():
        if k not in country_map:
            country_map[k] = v

    baci_df['i_clean'] = pd.to_numeric(baci_df['i'], errors='coerce')
    baci_df['country_name'] = baci_df['i_clean'].map(country_map)
    
    baci_df['country_name'] = baci_df['country_name'].fillna(
        baci_df['i_clean'].apply(lambda x: f"국가코드 {int(x)}" if pd.notnull(x) else "기타")
    )
    
    baci_df['무역액등급'] = pd.qcut(
        baci_df['v'], 
        q=3, 
        labels=['소', '중', '대']
    )
    
    return baci_df

df = load_data()

# 4. 사이드바 필터
st.sidebar.header("필터 설정")

all_countries = sorted(df['country_name'].unique().tolist())

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

# 5. 오른쪽 메인 화면 출력
# 1. 타이틀
st.title("무역 분석 대시보드")

# 2. baci_85_sample.csv 파일의 결측치
st.subheader("baci_85_sample.csv 파일의 결측치")
missing_df = df.isnull().sum().reset_index()
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

# 5. 상위 5개국 * 무역액 등급 교차표
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