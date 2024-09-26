import streamlit as st
import pandas as pd

traffic_data = pd.read_excel('../data/traffic_2024_07.xlsx', sheet_name='2024년 07월')# 판다스를 이용하여 excel 파일에서 '2024년 07월' 시트의 데이터 불러오기
st.title('서울시 교통 지점별 교통량 분석')# 제목 출력

# 서울시의 교통 지점별의 교통량을 분석하려고 합니다.
# 사용자의 편의성을 위해 서울시의 교통 지점들을을 streamlit의 multiselect 함수를 통해 선택할 수 있도록 합니다.
# 이때 traffic data에서 교통 지점들은 '지점명' column에 저장되어 있습니다.
# 참고 multiselect: https://docs.streamlit.io/develop/api-reference/widgets/st.multiselect
loc = traffic_data['지점명'].unique()  # 지점명들의 list
selected_locations = st.multiselect('확인할 지점들을 선택하세요', loc)


filtered_data = traffic_data[traffic_data['지점명'].isin(selected_locations)]# 선택한 지점의 데이터 필터링


# 날짜별 총 교통량 계산
if not filtered_data.empty:
    # 시간대별 교통량 합계 계산
    time_columns = [str(i) + '시' for i in range(24)]
    filtered_data['총교통량'] = filtered_data[time_columns].sum(axis=1)
    daily_traffic = filtered_data.groupby('일자')['총교통량'].sum()
    st.write('## 선택한 지점들의 날짜별 총 교통량')
    # 교통량을 시계열 그래프인 line_chart로 표시
    # 참고 line_chart: https://docs.streamlit.io/develop/api-reference/charts/st.line_chart
    st.line_chart(daily_traffic)

else:
    st.write('선택한 지점에 해당하는 데이터가 없습니다.')
