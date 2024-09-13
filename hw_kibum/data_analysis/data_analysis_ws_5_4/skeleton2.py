import pandas as pd

# 2. 데이터 병합
purchase_data = ____________
satisfaction_data = ___________

# 고객 ID를 기준으로 데이터 병합
merged_data = pd.merge(________________)

# 병합 후 데이터셋 크기 확인
print(f"병합된 데이터셋 크기: {merged_data.shape}")
print(merged_data.head())

# 병합된 데이터를 CSV 파일로 저장
______________
