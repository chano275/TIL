import streamlit as st
import pandas as pd
import matplotlib as plt 

data = pd.read_excel('../data/traffic_2024_07.xlsx', sheet_name='2024년 07월')

st.title('서울시 2024년 7월 교통 데이터 시각화 앱')  # 제목 출력
st.write('## 데이터프레임 정보') 
st.write(data.info())                             # 데이터프레임 정보 출력
st.write('## 데이터 미리보기')
st.dataframe(data.head())                         # 데이터프레임 표시

# 슬라이더로 표시할 행 수 선택
row_count = st.slider('표시할 행 수를 선택하세요', min_value=1, max_value=100, value=5)
st.write(f'## 상위 {row_count}개 데이터')          # 선택한 행 수만큼 데이터 표시
st.dataframe(data.head(row_count))

#####

st.title('서울시 2024년 7월 교통 데이터 분석 앱')  # 제목 출력

# 선택할 열 목록 (시간대별 교통량 컬럼들)
time_columns = [str(i) + '시' for i in range(24)]

# SELECTBOX => 사용자가 열 선택
selected_column = st.selectbox('분석할 시간대를 선택하세요', time_columns)

# 선택한 열의 히스토그램 그리기
st.write(f'## {selected_column} 교통량 히스토그램')
fig, ax = plt.subplots()
ax.hist(data[selected_column].dropna(), bins=20)
st.pyplot(fig)
st.write(f'## {selected_column} 교통량 통계 정보')
st.write(data[selected_column].describe())  # 선택한 열의 통계 정보 표시

#####

data = pd.read_excel('/Users/jeongdohyeon/Desktop/saffy/문제_수정가능/0926_Data_11/practice_4/data/traffic_2024_07.xlsx', sheet_name='수집지점 주소 및 좌표')

# 컬럼명 영어로 변경
data.rename(columns={'위도': 'latitude', '경도': 'longitude'}, inplace=True)

st.title('서울시 교통 지점 위치 지도 시각화')  # 제목 출력
map_data = data[['latitude', 'longitude']].dropna()  # 위도와 경도 정보를 가진 데이터프레임 생성
st.write('## 교통 지점 위치')
st.map(map_data)  # 지도에 데이터 표시

#####

# 데이터 불러오기
traffic_data = pd.read_excel('/Users/jeongdohyeon/Desktop/saffy/문제_수정가능/0926_Data_11/practice_5/data/traffic_2024_07.xlsx', sheet_name='2024년 07월')
location_data = pd.read_excel('/Users/jeongdohyeon/Desktop/saffy/문제_수정가능/0926_Data_11/practice_5/data/traffic_2024_07.xlsx', sheet_name='수집지점 주소 및 좌표')

st.title('서울시 교통량 인터랙티브 시각화')

# 날짜 목록 생성
dates = sorted(traffic_data['일자'].unique())
selected_date = st.selectbox('날짜를 선택하세요', dates)

# 지점명 목록 생성
locations = traffic_data['지점명'].unique()
selected_locations = st.multiselect('지점명을 선택하세요', locations, default=locations[0])

# 선택한 날짜와 지점명으로 데이터 필터링
filtered_data = traffic_data[(traffic_data['일자'] == selected_date) & (traffic_data['지점명'].isin(selected_locations))]

# 시간대별 교통량 컬럼 선택
time_columns = [str(i) + '시' for i in range(24)]

# 필터링된 데이터의 시간대별 교통량 합계 계산
if not filtered_data.empty:
    traffic_sum = filtered_data[time_columns].sum()
    st.write('## 선택한 지점들의 시간대별 교통량 합계')
    st.line_chart(traffic_sum)
else:st.write('선택한 조건에 해당하는 데이터가 없습니다.')

##### 

traffic_data = pd.read_excel('/Users/jeongdohyeon/Desktop/saffy/문제_수정가능/0926_Data_11/assignment_1/data/traffic_2024_07.xlsx', sheet_name='2024년 07월')
location_data = pd.read_excel('/Users/jeongdohyeon/Desktop/saffy/문제_수정가능/0926_Data_11/assignment_2/data/traffic_2024_07.xlsx', sheet_name='수집지점 주소 및 좌표')

# '일자' 열을 날짜 형식으로 변환
traffic_data['일자'] = pd.to_datetime(traffic_data['일자'], format='%Y%m%d')

st.title('서울시 교통 지점별 교통량 분석')

# 지점명 목록 생성
locations = traffic_data['지점명'].unique()
selected_locations = st.multiselect('교통 지점을 선택하세요', locations, default=[locations[0]])

# 선택한 지점의 데이터 필터링
filtered_data = traffic_data[traffic_data['지점명'].isin(selected_locations)]

# 날짜별 총 교통량 계산
if not filtered_data.empty:
    # 시간대별 교통량 합계 계산
    time_columns = [str(i) + '시' for i in range(24)]
    filtered_data['총교통량'] = filtered_data[time_columns].sum(axis=1)
    daily_traffic = filtered_data.groupby('일자')['총교통량'].sum()

    # 교통량 시계열 그래프 표시
    st.write('## 선택한 지점들의 날짜별 총 교통량')
    st.line_chart(daily_traffic)
else:st.write('선택한 지점에 해당하는 데이터가 없습니다.')


##### 

st.title('서울시 교통 데이터 시각화 대시보드')

# 사이드바에 필터 옵션 추가
st.sidebar.header('필터 옵션')

# 지점명 선택
locations = traffic_data['지점명'].unique()
selected_locations = st.sidebar.multiselect('교통 지점을 선택하세요', locations, default=locations)

# 날짜 범위 선택
dates = sorted(traffic_data['일자'].unique())
start_date = st.sidebar.selectbox('시작 날짜', dates, index=0)
end_date = st.sidebar.selectbox('종료 날짜', dates, index=len(dates)-1)

# 데이터 필터링
filtered_data = traffic_data[
    (traffic_data['지점명'].isin(selected_locations)) &
    (traffic_data['일자'] >= start_date) &
    (traffic_data['일자'] <= end_date)
]
st.write('## 필터링된 데이터')
st.dataframe(filtered_data.head())

# 시간대별 교통량 합계 계산
if not filtered_data.empty:
    time_columns = [str(i) + '시' for i in range(24)]
    filtered_data['총교통량'] = filtered_data[time_columns].sum(axis=1)

    # 날짜별 총 교통량 시계열 그래프
    daily_traffic = filtered_data.groupby('일자')['총교통량'].sum()
    st.write('## 날짜별 총 교통량')
    st.line_chart(daily_traffic)

    # 선택한 지점들의 위치 지도 시각화
    st.write('## 선택한 지점들의 위치')

    # '지점명칭'을 사용하여 필터링 + '위도'와 '경도' 컬럼명을 영어로 변경
    selected_locations_data = location_data[location_data['지점명칭'].isin(selected_locations)]
    selected_locations_data = selected_locations_data.rename(columns={'위도': 'latitude', '경도': 'longitude'})

    # 지도에 사용할 데이터 생성
    map_data = selected_locations_data[['latitude', 'longitude']].dropna()

    if not map_data.empty:st.map(map_data)
    else:                 st.write('지도에 표시할 위치 정보가 없습니다.')

else:
    st.write('선택한 조건에 해당하는 데이터가 없습니다.')
