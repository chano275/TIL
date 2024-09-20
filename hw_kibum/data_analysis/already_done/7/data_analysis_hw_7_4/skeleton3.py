import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
file_path = 'C:/Users/SSAFY/Desktop/TIL/hw_kibum/data_analysis/data_analysis_hw_7_4/time_series_data.csv'
data = pd.read_csv(file_path, parse_dates=['Date'])

# 특정 기간 필터링
filtered_data = data[(data['Date'] >= '2023-01-01') & (data['Date'] <= '2023-06-30')]

# 시간에 따른 변화 시각화
plt.figure(figsize=(12, 6))
plt.plot(filtered_data['Date'], filtered_data['value'], marker='o')
plt.title('Time Series Data (2023 H1)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.show()

# 특정 기간 동안의 평균값과 최대값 계산
mean_value = filtered_data['value'].mean()
max_value = filtered_data['value'].max()

print(f"2023년 상반기 평균값: {mean_value:.2f}")
print(f"2023년 상반기 최대값: {max_value:.2f}")
