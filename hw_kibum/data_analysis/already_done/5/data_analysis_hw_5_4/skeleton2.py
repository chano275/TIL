import pandas as pd

# 병합된 데이터 불러오기
merged_data = pd.read_csv('C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_hw_5_4/merged_data.csv')

print(merged_data.head())

# 새로운 피처 생성
merged_data['IncomePerPurchase'] = merged_data['AnnualIncome'] / merged_data['PurchaseAmount']
merged_data['AgeIncomeRatio'] = merged_data['Age'] / merged_data['AnnualIncome']

# 생성된 피처 확인
print(merged_data[['IncomePerPurchase', 'AgeIncomeRatio']].head())

# 피처 엔지니어링 후 데이터 저장
merged_data.to_csv('C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_hw_5_4/engineered_data.csv', index=False)
