import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 데이터 로드
df = pd.read_csv('____________')  # 파일 경로 입력
X = df[['____________']]  # 독립변수 설정
y = df['____________']  # 종속변수 설정

# 단순 선형 회귀 모델 구축
model = LinearRegression()
model.fit(X, y)

# 계수와 절편 출력
coefficient = model.coef_[0]
intercept = model.intercept_
print(f"문제 1. 단순 선형 회귀 모델의 계수(Coefficient): {coefficient}")
print(f"문제 1. 단순 선형 회귀 모델의 절편(Intercept): {intercept}")

# 모델 성능 평가 (R-squared)
r_squared = model.score(X, y)
print(f"문제 2. 모델의 R-squared 값: {r_squared}")

# 시각화 - 실제 값과 회귀선
plt.figure(figsize=(10, 6))
plt.scatter(df['____________'], df['____________'], alpha=0.5, label='Actual Data')  # X, y 설정
plt.plot(df['____________'], model.predict(X), color='red', label='Regression Line')  # X 설정
plt.title('Traffic Volume vs. Average Speed with Regression Line')
plt.xlabel('____________')  # x축 레이블 설정
plt.ylabel('____________')  # y축 레이블 설정
plt.legend()
plt.grid(True)
plt.show()

# 결과 해석
print("문제 3. 해석: 회귀 계수는 ____________로, 교통량이 증가할수록 평균 속도가 ____하는 경향을 나타냅니다.")  # 빈칸 채우기
print("절편은 _______로, 교통량이 0일 때 예상되는 평균 속도입니다.")  # 빈칸 채우기
print("R-squared 값이 _______이라는 것은 이 모델이 _____ 데이터를 설명할 수 있다는 뜻입니다. 이는 모델이 교통량과 속도 간의 관계를 어느 정도 설명하지만, 완벽하지 않다는 것을 보여줍니다.")  # 빈칸 채우기
