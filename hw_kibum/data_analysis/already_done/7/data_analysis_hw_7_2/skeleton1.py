import pandas as pd

# 데이터 로드
file_path = 'C:/Users/SSAFY/Desktop/TIL/hw_kibum/data_analysis/data_analysis_hw_7_2/group_data.csv'
data = pd.read_csv(file_path)

# 특정 조건을 만족하는 데이터 필터링 (예: 특정 열 값이 10에서 20 사이)
filtered_data = data[(data['value'] >= 10) & (data['value'] <= 20)]

# 필터링된 데이터 확인
print(filtered_data.head())
