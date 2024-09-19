import pandas as pd

# 데이터 불러오기
demo_data = pd.read_csv('C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_hw_5_4/demo_data.csv')
purchase_data = pd.read_csv('C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_hw_5_4/purchase_data.csv')

# 데이터 병합
merged_data = pd.merge(demo_data, purchase_data, on='CustomerID')

# 병합된 데이터 확인
print("병합된 데이터의 크기:", merged_data.shape)
print(merged_data.head())

# 병합된 데이터 저장
merged_data.to_csv('C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_hw_5_4/merged_data.csv', index=False)
