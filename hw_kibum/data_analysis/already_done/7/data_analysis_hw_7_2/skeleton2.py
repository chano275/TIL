import pandas as pd

# 데이터 로드
file_path = 'C:/Users/SSAFY/Desktop/TIL/hw_kibum/data_analysis/data_analysis_hw_7_2/group_data.csv'
data = pd.read_csv(file_path)

# 특정 조건을 만족하는 데이터 필터링 (예: 특정 열 값이 10에서 20 사이)
filtered_data = data[(data['value'] >= 10) & (data['value'] <= 20)]

# 특정 열을 기준으로 그룹화하고 요약 통계 계산 (예: 'category' 열 기준)
grouped_data = filtered_data.groupby('category').describe()

# 그룹별 요약 통계 확인
print(grouped_data)
