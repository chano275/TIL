from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('____________')  # 'seoul_traffic_data.csv' 경로 입력

# 독립 변수와 종속 변수 설정
X = df[['____________']]  # 'traffic_volume'을 입력하여 교통량 데이터를 독립 변수로 설정합니다.
y = df['____________']    # 'avg_speed'를 입력하여 평균 속도를 종속 변수로 설정합니다.

# 문제 1. 산점도 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5)  # 데이터 산점도를 그려 데이터의 분포를 확인합니다.
plt.title('Traffic Volume vs. Average Speed')
plt.xlabel('____________')  # x축 라벨을 작성하세요
plt.ylabel('____________')  # y축 라벨을 작성하세요
plt.grid(True)  # 그리드를 추가하여 그래프를 보기 쉽게 만듭니다.
plt.show()  # 그래프를 화면에 출력합니다.

# 문제 2. 선형 회귀 모델 구축
linear_model = LinearRegression()
linear_model.fit(____________, __________)  # 선형 회귀 모델을 독립 변수와 종속 변수로 학습시킵니다.
linear_r2 = linear_model.score(____________, __________)  # 선형 회귀 모델의 R-squared 값을 계산하여 모델 성능을 평가합니다.
print(f"Linear Model R-squared: {linear_r2}")

# 문제 2. 2차 다항 회귀 모델 구축
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(____________)  # X 데이터를 2차 다항식 형태로 변환합니다.
poly_model = LinearRegression()
poly_model.fit(____________, __________)  # 다항 회귀 모델을 변환된 데이터와 종속 변수로 학습시킵니다.
poly_r2 = poly_model.score(____________, __________)  # 다항 회귀 모델의 R-squared 값을 계산하여 성능을 평가합니다.
print(f"Polynomial Model (Degree 2) R-squared: {poly_r2}")

# 문제 3. 시각화 - 선형 회귀와 2차 다항 회귀 비교
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data')  # 실제 데이터를 산점도로 표시합니다.
plt.plot(X, linear_model.predict(X), color='red', label='Linear Regression')  # 선형 회귀선 추가
plt.plot(X, poly_model.predict(X_poly), color='blue', label='Polynomial Regression (Degree 2)')  # 다항 회귀선 추가
plt.title('Traffic Volume vs. Average Speed with Regression Lines')
plt.xlabel('____________')  # x축 라벨을 작성하세요
plt.ylabel('____________')  # y축 라벨을 작성하세요
plt.legend()
plt.grid(True)
plt.show()

# 문제 4. 결과 해석
print("\n문제 4. 해석:")
print("선형 회귀 모델과 다항 회귀 모델을 비교한 결과, 다항 회귀 모델의 R-squared 값이 더 ___ 나왔습니다.")  # '높게' 또는 '낮게'와 같은 단어를 넣어 모델 성능을 비교하세요.
print("이는 다항 회귀 모델이 선형 모델보다 데이터의 비선형 관계를 더 _________것을 의미합니다.")  # '잘 설명하는' 등의 설명을 통해 비선형 관계 설명을 강조합니다.
