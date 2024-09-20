import pandas as pd

# 데이터 로드
file_path = 'C:/Users/SSAFY/Desktop/TIL/hw_kibum/data_analysis/data_analysis_hw_7_4/time_series_data.csv'
data = pd.read_csv(file_path)

# 데이터 기본 정보 확인
print(data.info())
print(data.describe())
