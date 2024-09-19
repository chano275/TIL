import pandas as pd

# 1. 데이터 로드 및 확인
file_path = 'C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_ws_4_2/survey_data.csv'
data = pd.read_csv(file_path)

# 데이터 확인
print(data.head())
print(data.describe())
