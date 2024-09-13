import pandas as pd

# 1. 데이터 로드
file_path = 'C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_ws_4_1/seoul_subway_data.csv'
data = pd.read_csv(file_path)

# 2. 요일별 및 시간대별 이용객 수 집계
# groupby로 요일 / 시간에 따라 나눈 후, 특정 열을 잡고 합계를 구한다.
weekday_agg = data.groupby('DayOfWeek')['PassengerCount'].sum()
time_agg = data.groupby('Time')['PassengerCount'].sum()

# # 결과 확인
print(weekday_agg)
print(time_agg)
