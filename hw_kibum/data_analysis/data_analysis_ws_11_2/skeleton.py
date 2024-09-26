import streamlit as st
import pandas as pd

# 데이터 불러오기
data = pd.read_excel('_________', sheet_name='_________')

# 제목 출력
st.title('_________')

# 슬라이더로 표시할 행 수 선택
row_count = st.slider('표시할 행 수를 선택하세요', min_value=_________, max_value=_________, value=_________)

# 선택한 행 수만큼 데이터 표시
st.write(f'## 상위 {_________}개 데이터')
st.dataframe(data.head(_________))
