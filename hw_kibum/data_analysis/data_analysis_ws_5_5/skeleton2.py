import pandas as pd

# 2. 이상치 처리
file_path = _____________
sales_data = ____________

# IQR 방법을 사용하여 이상치 제거
Q1 = sales_data.quantile(0.25)
Q3 = sales_data.quantile(0.75)
IQR = Q3 - Q1

# IQR 범위 밖의 이상치 제거
filtered_data = ________________

# 처리된 데이터 저장
filtered_data.to_csv('../data/filtered_sales_data.csv', index=False)
print(f"처리된 데이터셋 크기: {filtered_data.shape}")
