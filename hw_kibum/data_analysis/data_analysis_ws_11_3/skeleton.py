import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_excel('_________', sheet_name='_________')

# 제목 출력
st.title('_________')

# 선택할 열 목록 (시간대별 교통량 컬럼들)
time_columns = [_________ for i in range(_________)]

# 사용자가 열 선택
selected_column = st.selectbox('분석할 시간대를 선택하세요', _________)

# 선택한 열의 히스토그램 그리기
st.write(f'## {_________} 교통량 히스토그램')
fig, ax = plt.subplots()
ax.hist(data[_________].dropna(), bins=_________)
st.pyplot(fig)

# 선택한 열의 통계 정보 표시
st.write(f'## {_________} 교통량 통계 정보')
st.write(data[_________].describe())
