import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# 데이터 로드
df = pd.read_csv('seoul_traffic_data.csv')  # 파일 경로를 적절히 입력하세요.

# 날짜 데이터를 숫자형 데이터로 변환 후 원래 날짜 컬럼 삭제
df['date'] = pd.to_datetime(df['date'])  # 날짜 데이터를 datetime 형식으로 변환
df['year'] = df['date'].dt.year  # 연도 정보 추출
df['month'] = df['date'].dt.month  # 월 정보 추출
df['day'] = df['date'].dt.day  # 일 정보 추출
df = df.drop('date', axis=1)  # 원래 날짜 컬럼 삭제

# 범주형 데이터 처리 - One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

# 문제 1: 이상치 제거
upper_bound = df['traffic_volume'].quantile(0.99)  # quantile < 하위 0.@@ % >>> 상위 1%를 이상치로 간주
df_cleaned = df[df['traffic_volume'] <= upper_bound]
print("문제 1. 이상치 제거 후 데이터:\n", df_cleaned.head())  # 이상치 제거 후 데이터를 출력하세요.

# 문제 2: 회귀 모델 성능 비교
X = df.drop(['traffic_volume'], axis=1)  # 타겟 변수 제거
y = df['traffic_volume']  # 타겟 변수 설정
X_cleaned = df_cleaned.drop(['traffic_volume'], axis=1)  # 타겟 변수 제거
y_cleaned = df_cleaned['traffic_volume']  # 타겟 변수 설정

# 기존 데이터로 모델 훈련
model = LinearRegression()
model.fit(X, y)  # 기존 데이터로 모델 훈련
r_squared_original = model.score(X, y)  # 기존 데이터의 R-squared 계산

# 이상치 제거 데이터로 모델 훈련
model.fit(X_cleaned, y_cleaned)  # 이상치 제거 데이터로 모델 훈련
r_squared_cleaned = model.score(X_cleaned, y_cleaned)  # 이상치 제거 데이터의 R-squared 계산

print(f"문제 2. 원본 데이터 R-squared: {r_squared_original}")
print(f"문제 2. 이상치 제거 데이터 R-squared: {r_squared_cleaned}")

# 결과 해석
if r_squared_cleaned > r_squared_original:
    print("해석: 이상치를 제거했을 때 R-squared 값이 상승하여 모델이 데이터를 더 잘 설명하게 되었습니다.")
else:
    print("해석: 이상치를 제거했지만 R-squared 값이 하락하거나 거의 변화가 없었습니다. 이는 이상치 제거가 항상 모델 성능을 개선하지 않는다는 것을 보여줍니다.")

# 문제 3: 교차 검증
cross_val_scores = cross_val_score(model, X_cleaned, y_cleaned, cv=5)  # 5-fold 교차 검증을 수행하여 모델의 일관된 성능을 평가합니다.
print(f"문제 3. 평균 R-squared from 5-fold CV: {cross_val_scores.mean()}")

# 주석 설명: 5-fold CV는 데이터를 5개로 나누어 각기 다른 부분을 테스트하고 나머지로 훈련하는 과정을 반복해 모델의 일관된 성능을 평가합니다.
