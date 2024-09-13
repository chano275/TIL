import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 로드
file_path = 'C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_ws_4_1/seoul_subway_data.csv'
data = pd.read_csv(file_path)

# 2. 요일별 이용객 수 집계
weekday_agg = data.groupby('DayOfWeek')['PassengerCount'].sum()

# 3. 요일별 이용객 수 가로 막대 그래프 시각화
plt.figure()  # figsize=(10, 6)
weekday_agg.plot(kind='barh')  # weekday_agg.sort_values().plot(kind='barh', color='skyblue')
plt.title('Total Passengers by Day of the Week')
plt.xlabel('Passenger Count')
plt.ylabel('Day of the Week')
plt.show()
