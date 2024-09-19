import pandas as pd

# 데이터 불러오기
file_path = ('C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_hw_5_2/customer_data.csv')
data = pd.read_csv(file_path)

# 결측치 탐지
print("결측치 확인:")
print(pd.isnull(data))
print(data.isnull().sum())