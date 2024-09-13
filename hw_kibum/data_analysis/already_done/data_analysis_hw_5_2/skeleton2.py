import pandas as pd

# 데이터 불러오기
file_path = ('C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_hw_5_2/customer_data.csv')
data = pd.read_csv(file_path)

# 결측치 처리 (평균값 또는 중앙값으로 대체) 
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['AnnualIncome'] = data['AnnualIncome'].fillna(data['AnnualIncome'].mean())
data['SpendingScore'] = data['SpendingScore'].fillna(data['SpendingScore'].mean())
data['PurchaseHistory'] = data['PurchaseHistory'].fillna(data['PurchaseHistory'].mean())

# 결측치 처리 후 데이터 저장
data.to_csv('C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_hw_5_2/processed_data.csv', index=False)
