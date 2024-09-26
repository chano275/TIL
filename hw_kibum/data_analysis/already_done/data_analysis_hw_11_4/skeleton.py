import streamlit as st
import pandas as pd

# 판다스를 이용하여 excel 파일에서 '2024년 07월'과 '수집지점 주소 및 좌표'시트의 데이터룰 불러오기
# 각각 traffic_data 와 location_data 변수로 불러오기
traffic_data = pd.read_excel('../data/traffic_2024_07.xlsx', sheet_name='2024년 07월')
location_data = pd.read_excel('../data/traffic_2024_07.xlsx', sheet_name='수집지점 주소 및 좌표')

# streamlit의 title함수를 이용하여 제목('서울시 교통 데이터 시각화 대시보드') 출력
st.title('서울시 교통 데이터 시각화 대시보드')

# 더 편한 접근성을 위해 sidebar를 사용하여 필터 옵션을 제공합니다.
# 참고 : https://docs.streamlit.io/develop/api-reference/layout/st.sidebar
st.sidebar.header('필터 옵션')  # 이때, 사이드바의 제목(header)은 '필터 옵션'으로 설정합니다.

# 서울시의 교통 지점별의 교통량을 분석하려고 합니다.
# 사용자의 편의성을 위해 서울시의 교통 지점들을을 streamlit의 multiselect 함수를 통해 선택할 수 있도록 합니다.
# 이때 traffic data에서 교통 지점들은 '지점명' column에 저장되어 있습니다.
# 참고: https://docs.streamlit.io/develop/api-reference/widgets/st.multiselect
loc = traffic_data['지점명'].unique()  # 지점명들의 list
selected_locations = st.multiselect('교통 지점들을 선택하세요', loc)

# 날짜 범위 선택하는 코드를 작성합니다 날짜를 선택할 수 있는 방법은 여러가지가 존재하겠지만, 이 실습에서는
# 사용자가 시작 날짜와 종료 날짜를 선택할 수 있도록 st.sidebar.selectbox()를 사용해보겠습니다
# selectbox는 '시작 날짜'와 '종료 날짜'를 선택할 수 있도록 만들어줍니다.
# 시작 날짜는 dates의 첫번째 값으로, 종료 날짜는 dates의 마지막 값으로 설정합니다.
# 참고 : https://docs.streamlit.io/develop/api-reference/widgets/st.selectbox
dates = sorted(traffic_data['일자'].unique())  # 날짜 범위는 traffic_data의 '일자' column에 저장되어 있습니다.
start = st.sidebar.selectbox('시작일', dates, index=0)
end = st.sidebar.selectbox('종료일', dates, index=len(dates) - 1)

# 이후 해당하는 지점을 기준으로, 선택한 날짜 범위에 해당하는 데이터를 필터링합니다.
filtered = traffic_data[(traffic_data['지점명'].isin(selected_locations)) & (traffic_data['일자'] >= start) & (traffic_data['일자'] <= end)]
st.write('## 필터링된 데이터')

# 필터링된 데이터를 st.dataframe()을 이용하여 화면에 출력합니다.
# 참고 : https://docs.streamlit.io/develop/api-reference/elements/st.dataframe
st.dataframe(filtered.head())

if filtered.empty:
    st.write('선택한 조건의 데이터가 없습니다.')
else:
    # 필터링된 데이터의 시간대별 교통량 합계 계산
    # 현재 traffic_data의 시간대를 나타내는 column 명들이 '0시', '1시', '2시'등으로 표현되어있기 때문에 이를 고려하여 교통량 컬럼 선택하고
    # 교통량을 시계열 그래프인 line_chart로 표시
    # 참고 line_chart: https://docs.streamlit.io/develop/api-reference/charts/st.line_chart
    option = []
    for i in range(24):option.append(f'{i}시')
    filtered['총 교통량'] = filtered[option].sum(axis=1)  # 컬럼 추가하고
    daily = filtered.groupby('일자')['총 교통량'].sum()    # 일자별로 총 교통량 sum ... 고르는 장소와는 연관이 없는듯
    st.line_chart(daily)


    ### 선택한 지점들의 위치 지도 시각화
    # location_data에는 지점명칭과 위도, 경도 정보가 저장되어 있습니다.
    # 우리는 선택한 지점들의 위치를 지도에 표시하기 위해 location_data에서 선택한 지점들의 데이터만 가져옵니다.
    new_df = location_data[location_data['지점명칭'].isin(selected_locations)]
    # 이후 지도에 표시할 데이터중 위도와 경도가 있는 데이터만 추출합니다 (결측치 제거)
    map_df = new_df[['위도', '경도']].dropna()

    if map_df.empty:st.write('지도에 표시할 정보가 없습니다.')
    else:
        # 추출된 데이터는 지도에 streamlit의 map() 함수를 사용하여 표시합니다.
        # 참고: https://docs.streamlit.io/develop/api-reference/charts/st.map
        st.map(map_df, latitude='위도', longitude='경도')

        # 만약, streamlit의 map() 에서 추가적인 latitude와 longitude 컬럼명을 명시하지 않는다면
        # 'LAT', 'LATITUDE', 'lat', 'latitude' 컬럼이 존재하는지 확인하고 존재한다면 위도로 사용
        # 'LON', 'LONGITUDE', 'lon', 'longitude' 컬럼이 존재하는지 확인하고 존재한다면 경도로 사용
        # 만약 존재하지 않는다면 에러를 발생시킵니다.
        # 우리의 위도와 경도 컬럼명은 '위도'와 '경도'이므로 컬럼명을 파라미터로 전달하여 지도에 표시합니다.




