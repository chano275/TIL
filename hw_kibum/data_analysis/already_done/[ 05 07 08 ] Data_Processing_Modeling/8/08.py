import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 데이터 로드
df = pd.read_csv('../data/seoul_traffic_data.csv')

# 날짜 데이터 처리: datetime 변환 및 연, 월, 일 추출
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df = df.drop('date', axis=1)

# 범주형 데이터 원-핫 인코딩 처리
df = pd.get_dummies(df, drop_first=True)

# 이상치 제거: 교통량 상위 1% 제거
upper_bound = df['traffic_volume'].quantile(0.99)
df_cleaned = df[df['traffic_volume'] <= upper_bound]

# 데이터 탐색 및 파생 변수 생성
df_cleaned['traffic_density'] = df_cleaned['traffic_volume'] / df_cleaned['num_vehicles']  # 새로운 변수 생성
correlation = df_cleaned['traffic_volume'].corr(df_cleaned['temperature'])  # 상관관계 계산
print("교통량과 기온의 상관관계:", correlation)

# 단순 선형 회귀 모델: 교통량으로 평균 속도 예측
X_simple = df_cleaned[['traffic_volume']]
y_simple = df_cleaned['avg_speed']
model_simple = LinearRegression()
model_simple.fit(X_simple, y_simple)
r_squared_simple = model_simple.score(X_simple, y_simple)
print(f"단순 선형 회귀 모델의 R-squared 값: {r_squared_simple}")

# 회귀선 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X_simple, y_simple, alpha=0.5, label='실제 데이터')
plt.plot(X_simple, model_simple.predict(X_simple), color='red', label='선형 회귀선')
plt.title('교통량과 평균 속도의 관계')
plt.xlabel('교통량')
plt.ylabel('평균 속도')
plt.legend()
plt.grid(True)
plt.show()

# 2차 다항 회귀 모델 구축
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_simple)
model_poly = LinearRegression()
model_poly.fit(X_poly, y_simple)
r_squared_poly = model_poly.score(X_poly, y_simple)
print(f"2차 다항 회귀 모델의 R-squared 값: {r_squared_poly}")

# 선형 회귀와 다항 회귀 비교 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X_simple, y_simple, alpha=0.5, label='실제 데이터')
plt.plot(X_simple, model_simple.predict(X_simple), color='red', label='선형 회귀')
plt.plot(X_simple, model_poly.predict(X_poly), color='blue', label='2차 다항 회귀')
plt.title('교통량과 평균 속도의 관계 (회귀 비교)')
plt.xlabel('교통량')
plt.ylabel('평균 속도')
plt.legend()
plt.grid(True)
plt.show()

# 다중 선형 회귀 모델: 여러 특성으로 교통량 예측
X = df_cleaned.drop('traffic_volume', axis=1)
y = df_cleaned['traffic_volume']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"훈련 데이터셋 크기: {len(X_train)}개, 테스트 데이터셋 크기: {len(X_test)}개")

# 모델 훈련
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# 모델 평가
r_squared_multi = model_multi.score(X_test, y_test)
print(f"다중 선형 회귀 모델의 R-squared 값: {r_squared_multi}")

# 예측 및 평가 지표 계산
y_pred = model_multi.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"MSE: {mse}")
print(f"MAE: {mae}")

# 5-fold 교차 검증
cv_scores = cross_val_score(model_multi, X, y, cv=5)
print(f"5-fold 교차 검증 평균 R-squared 값: {cv_scores.mean()}")

# 변수의 계수 확인
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model_multi.coef_})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
print("다중 선형 회귀 모델의 상위 5개 변수와 계수:")
print(coefficients.head(5))

# 결과 해석
print("\n해석:")
print(f"단순 선형 회귀 R-squared: {r_squared_simple}")
print(f"2차 다항 회귀 R-squared: {r_squared_poly}")
print(f"다중 선형 회귀 R-squared: {r_squared_multi}")
print("다중 회귀 모델이 가장 높은 설명력을 가지며, 이는 여러 변수를 활용하여 교통량을 더 정확하게 예측할 수 있음을 의미합니다.")

"""
설명:
데이터 전처리: 날짜 데이터 처리와 범주형 데이터의 원-핫 인코딩, 그리고 이상치 제거를 한 번에 수행하였습니다.
단순 선형 회귀 모델: traffic_volume을 사용하여 avg_speed를 예측하는 모델을 구축하고 성능을 평가하였습니다.
다항 회귀 모델: 2차 다항 회귀를 적용하여 비선형 관계를 모델링하고 선형 회귀와 비교하였습니다.
다중 선형 회귀 모델: 여러 변수를 사용하여 traffic_volume을 예측하는 모델을 구축하고, 교차 검증을 통해 성능을 평가하였습니다.
결과 해석: 각 모델의 R-squared 값을 비교하여 모델의 성능을 해석
"""