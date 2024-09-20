import pandas as pd

# 데이터 로드
file_path = 'C:/Users/SSAFY/Desktop/TIL/hw_kibum/data_analysis/data_analysis_hw_7_4/time_series_data.csv'
data = pd.read_csv(file_path, parse_dates=['Date'])

print(data.head())

# 특정 기간(예: 2023년 상반기) 필터링
filtered_data = data[(data['Date'] >= '2023-01-01') & (data['Date'] <= '2023-06-30')]
## 중요 

# 필터링된 데이터 확인
print(filtered_data.head())
