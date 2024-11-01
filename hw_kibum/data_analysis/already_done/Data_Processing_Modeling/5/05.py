import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

file_path = '../data/customer_data.csv'
data = pd.read_csv(file_path)

print(data.isnull().sum())  # 결측치 탐지
# 결측치 처리 (평균값 또는 중앙값으로 대체)
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['AnnualIncome'] = data['AnnualIncome'].fillna(data['AnnualIncome'].median())
data['SpendingScore'] = data['SpendingScore'].fillna(data['SpendingScore'].mean())
data['PurchaseHistory'] = data['PurchaseHistory'].fillna(data['PurchaseHistory'].median())
data.to_csv('../data/processed_data.csv', index=False)

print("결측치 처리 후 데이터셋:")
data = pd.read_csv('../data/processed_data.csv')
print(data.isnull().sum())

#####

demo_data = pd.read_csv('../data/demo_data.csv')
purchase_data = pd.read_csv('../data/purchase_data.csv')

merged_data = pd.merge(demo_data, purchase_data, on='CustomerID', how='inner')
print("병합된 데이터의 크기:", merged_data.shape)
print(merged_data.head())
merged_data.to_csv('../data/merged_data.csv', index=False)
merged_data = pd.read_csv('../data/merged_data.csv')

# 새로운 피처 생성
merged_data['IncomePerPurchase'] = merged_data['AnnualIncome'] / merged_data['PurchaseAmount']
merged_data['AgeIncomeRatio'] = merged_data['Age'] / merged_data['AnnualIncome']

# 생성된 피처 확인 + 피처 엔지니어링 후 데이터 저장
print(merged_data[['IncomePerPurchase', 'AgeIncomeRatio']].head())  
merged_data.to_csv('../data/engineered_data.csv', index=False)


data = pd.read_csv('../data/engineered_data.csv')

# 예측 대상 및 피처 설정
X = data.drop(columns=['CustomerID', 'PurchaseAmount', 'PurchaseDate'])
y = data['PurchaseAmount']

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델을 사용한 피처 중요도 분석
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 피처 중요도 시각화
importances = model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title('Feature Importance')
plt.show()

# 모델 예측 및 성능 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Model Mean Squared Error: {mse}')


#####

file_path = '../data/customer_data.csv'
customer_data = pd.read_csv(file_path)
print(customer_data.info())
print(customer_data.describe())

# 결측치 처리
customer_data = customer_data.dropna()
print(f"결측치 제거 후 데이터셋 크기: {customer_data.shape}")

# 이상치 탐지 및 처리 - 이상치 탐지 위해 IQR 방법 사용
for column in ['Age', 'AnnualIncome']:
    Q1 = customer_data[column].quantile(0.25)
    Q3 = customer_data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    initial_shape = customer_data.shape
    customer_data = customer_data[(customer_data[column] >= lower_bound) & (customer_data[column] <= upper_bound)]
    print(f"{column} 이상치 처리 전후 데이터셋 크기: {initial_shape} -> {customer_data.shape}")

# 나이와 연간 소득 간의 관계 시각화
sns.scatterplot(x='Age', y='AnnualIncome', data=customer_data)
plt.title('Age vs Annual Income')
plt.show()

#####

file_path = '../data/campaign_data.csv'
campaign_data = pd.read_csv(file_path)
campaign_data = campaign_data.drop_duplicates()  # 중복데이터 제거

# 각 캠페인별 참여율과 평균 클릭률 계산
campaign_summary = campaign_data.groupby('CampaignID').agg(
    ParticipationRate=('Participation', 'mean'),
    AvgClickRate=('Clicks', 'mean')
).reset_index()
print(campaign_summary)

# 캠페인 성과 시각화
sns.barplot(x='CampaignID', y='ParticipationRate', data=campaign_summary)
plt.title('Campaign Participation Rates')
plt.show()

# 참여율이 높은 고객과 낮은 고객의 매출 분포 시각화
high_participation = campaign_data[campaign_data['Participation'] == 1]['Revenue']
low_participation = campaign_data[campaign_data['Participation'] == 0]['Revenue']

plt.figure(figsize=(10, 6))
sns.histplot(high_participation, color='blue', label='High Participation', kde=True)
sns.histplot(low_participation, color='red', label='Low Participation', kde=True)
plt.title('Revenue Distribution by Participation')
plt.legend()
plt.show()

#####

file_path = '../data/sales_data.csv'
sales_data = pd.read_csv(file_path)

# 지역별 매출 계산
sales_data['Revenue'] = sales_data['Price'] * sales_data['Quantity']
region_sales = sales_data.groupby('Region')['Revenue'].sum().reset_index()

# 지역별 매출 시각화
sns.barplot(x='Region', y='Revenue', data=region_sales)
plt.title('Revenue by Region')
plt.show()

# 제품군별 매출 계산
sales_data['Revenue'] = sales_data['Price'] * sales_data['Quantity']
product_sales = sales_data.groupby('Product')['Revenue'].sum().reset_index()

# 제품군별 매출 점유율 시각화
plt.pie(product_sales['Revenue'], labels=product_sales['Product'], autopct='%1.1f%%')
plt.title('Revenue Share by Product')
plt.show()


# 월별 매출 시계열 분석
file_path = '../data/sales_data.csv'
sales_data = pd.read_csv(file_path)
# 날짜 데이터를 datetime 형식으로 변환
sales_data['Date'] = pd.to_datetime(sales_data['Date'])

# 매출 (Revenue) 열 계산
sales_data['Revenue'] = sales_data['Price'] * sales_data['Quantity']

# 월별 매출 계산
sales_data['Month'] = sales_data['Date'].dt.to_period('M').astype(str)  # 문자열로 변환
monthly_sales = sales_data.groupby('Month')['Revenue'].sum().reset_index()

# 월별 매출 시각화
plt.figure(figsize=(10, 6))
sns.lineplot(x='Month', y='Revenue', data=monthly_sales)
plt.title('Monthly Revenue Trend')
plt.xticks(rotation=45)
plt.show()


# 제품 가격과 판매량 간의 상관관계 시각화
file_path = '../data/sales_data.csv'
sales_data = pd.read_csv(file_path)

# 가격과 판매량 간의 상관관계 시각화
sns.scatterplot(x='Price', y='Quantity', data=sales_data)
plt.title('Price vs Quantity Sold')
plt.show()

#####

purchase_data = pd.read_csv('../data/purchase_history.csv')
satisfaction_data = pd.read_csv('../data/satisfaction_survey.csv')
print(purchase_data.info())
print(satisfaction_data.info())
merged_data = pd.merge(purchase_data, satisfaction_data, on='CustomerID', how='inner')  # 고객 ID를 기준으로 데이터 병합
print(f"병합된 데이터셋 크기: {merged_data.shape}")
print(merged_data.head())
merged_data.to_csv('../data/merged_data.csv', index=False)

# 병합된 데이터 로드 및 구매 횟수와 만족도 사이의 상관관계 시각화
merged_data = pd.read_csv('../data/merged_data.csv')

# 산점도로 시각화
sns.scatterplot(x='PurchaseHistory', y='Satisfaction', data=merged_data)
plt.title('Purchase History vs Satisfaction')
plt.show()

# 만족도와 구매 횟수를 기준으로 고객 세분화 시각화
sns.scatterplot(x='PurchaseHistory', y='Satisfaction', data=merged_data, hue='Satisfaction', size='TotalSpent', sizes=(20, 200))
plt.title('Customer Segmentation by Satisfaction and Purchase History')
plt.show()

#####

file_path = '../data/sales_data.csv'
sales_data = pd.read_csv(file_path)

# 이상치 탐지를 위한 박스플롯 시각화
plt.figure(figsize=(10, 6))
sns.boxplot(data=sales_data[['Price', 'SalesVolume', 'DiscountRate', 'AdvertisingExpense']])
plt.title('Boxplot of Features')
plt.show()

# IQR 방법을 사용하여 이상치 제거
Q1, Q3 = sales_data.quantile(0.25), sales_data.quantile(0.75)
IQR = Q3 - Q1

# IQR 범위 밖의 이상치 제거
filtered_data = sales_data[~((sales_data < (Q1 - 1.5 * IQR)) |(sales_data > (Q3 + 1.5 * IQR))).any(axis=1)]

filtered_data.to_csv('../data/filtered_sales_data.csv', index=False)
print(f"처리된 데이터셋 크기: {filtered_data.shape}")



filtered_data = pd.read_csv('../data/filtered_sales_data.csv')

# 새로운 피처 생성: Price per Advertising (가격 대비 광고비)와 DiscountedPrice (할인된 가격)
filtered_data['PricePerAd'] = filtered_data['Price'] / filtered_data['AdvertisingExpense']
filtered_data['DiscountedPrice'] = filtered_data['Price'] * (1 - filtered_data['DiscountRate'])
filtered_data.to_csv('../data/engineered_sales_data.csv', index=False)
print(filtered_data.head())



# 피처 중요도 분석
data = pd.read_csv('../data/engineered_sales_data.csv')

# 예측 대상 및 피처 설정
X = data.drop(columns=['ProductID', 'SalesVolume'])
y = data['SalesVolume']

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델을 사용한 피처 중요도 분석
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 피처 중요도 시각화
importances = model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title('Feature Importance')
plt.show()

# 모델 성능 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Model Mean Squared Error: {mse}')

#####