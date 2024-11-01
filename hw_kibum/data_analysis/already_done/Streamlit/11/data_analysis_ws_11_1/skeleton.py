import streamlit as st
import pandas as pd

# 데이터 불러오기
data = pd.read_excel('_________', sheet_name='_________')

# 제목 출력
st.title('_________')

# 데이터프레임 정보 출력
st.write('## 데이터프레임 정보')
st.write(_________)

# 데이터프레임 표시
st.write('## 데이터 미리보기')
st.dataframe(_________)
