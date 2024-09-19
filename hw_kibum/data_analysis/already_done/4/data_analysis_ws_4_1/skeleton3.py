import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 로드
file_path = 'C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_ws_4_1/seoul_subway_data.csv'
data = pd.read_csv(file_path)

# 2. 시간대별 이용객 수 집계
time_agg = data.groupby('Time')['PassengerCount'].sum()

# 3. 시간대별 이용객 수 꺾은선 그래프 시각화
plt.figure()
time_agg.plot(kind='line')  # time_agg.plot(kind='line', color='blue', marker='o')
plt.title('Total Passengers by Time of Day')
plt.xlabel('Time')
plt.ylabel('Passenger Count')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
