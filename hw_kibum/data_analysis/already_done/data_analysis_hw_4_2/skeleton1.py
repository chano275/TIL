import pandas as pd

# 1. 데이터 로드
file_path = 'C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_hw_4_2/category_data.csv'
data = pd.read_csv(file_path)

# 데이터 확인
print(data.head())
print(data.describe())

