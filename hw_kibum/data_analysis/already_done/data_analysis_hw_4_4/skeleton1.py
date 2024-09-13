import pandas as pd

# 1. 데이터 로드 및 확인
file_path = 'C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_hw_4_4/statistical_data.csv'
data = pd.read_csv(file_path)
print(data.head())
print(data.describe())
