import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드
file_path = ___________
customer_data = __________

# 데이터 구조 확인
print(customer_data.info())
print(customer_data.describe())

# 2. 결측치 처리
customer_data = _______________
print(f"결측치 제거 후 데이터셋 크기: {customer_data.shape}")

# 3. 이상치 탐지 및 처리
# 이상치를 탐지하기 위해 IQR 방법 사용
for column in ['Age', 'AnnualIncome']:
    Q1 = __________
    Q3 = __________
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    initial_shape = customer_data.shape
    customer_data = customer_data[(customer_data[column] >= lower_bound) & (customer_data[column] <= upper_bound)]
    print(f"{column} 이상치 처리 전후 데이터셋 크기: {initial_shape} -> {customer_data.shape}")

# 4. 나이와 연간 소득 간의 관계 시각화
_____________________
plt.title('Age vs Annual Income')
plt.show()
