import streamlit as st
import pandas as pd

# 위치 데이터 불러오기
data = pd.read_excel('_________', sheet_name='_________')

# 컬럼명 영어로 변경
data.rename(columns={'위도': '_________', '경도': '_________'}, inplace=True)

# 제목 출력
st.title('_________')

# 위도와 경도 정보를 가진 데이터프레임 생성
map_data = data[['_________', '_________']].dropna()

# 지도에 데이터 표시
st.write('## 교통 지점 위치')
st.map(_________)
