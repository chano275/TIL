import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error

# ----------------- 단순 선형 회귀 -----------------

weekdays_data = pd.read_excel('./data/weekday_traffic.xlsx')

# 독립 변수: 각 날의 인덱스 / 종속 변수: 각 날의 8시 교통량
days = np.arange(len(weekdays_data)).reshape(-1, 1)
traffic_at_8am = weekdays_data['8시'].values

# 데이터 분리 (훈련 80%, 테스트 20%)
X_train, X_test, y_train, y_test = train_test_split(days, traffic_at_8am, test_size=0.2, random_state=42)

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 마지막 날의 8시 교통량 예측
predicted_traffic = model.predict(np.array([[len(weekdays_data)]]))

# 실제 8시 교통량 가져오기
last_day_traffic = weekdays_data.iloc[-1]['8시']

# 결과 출력
print(f"실제 교통량: {last_day_traffic:.2f}, 예측 교통량: {predicted_traffic[0]:.2f}")

# ----------------- 랜덤 포레스트 분류 -----------------

# 데이터 로드
df = pd.read_excel('./data/weekday_traffic.xlsx')

# 독립 변수와 종속 변수 설정
X = df.loc[:, '0시':'23시']
y = df['혼잡']

# 데이터 분리 (훈련 70%, 테스트 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 랜덤 포레스트 모델 생성 및 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 테스트 데이터의 첫 번째 샘플 선택
X_first = X_test.iloc[[0]]

# 해당 샘플의 날짜와 실제 혼잡 여부 가져오기
date_first = df.loc[X_test.index[0], '일자']
actual_congestion = y_test.iloc[0]

# 혼잡 여부 예측
predicted_congestion = model.predict(X_first)[0]

# 결과 출력
print(f"선택된 날짜: {date_first}")
print(f"실제 혼잡 여부: {'혼잡' if actual_congestion else '비혼잡'}")
print(f"예측 혼잡 여부: {'혼잡' if predicted_congestion else '비혼잡'}")

# ----------------- KNN 분류 및 성능 평가 -----------------

# 데이터 로드
df = pd.read_excel('./data/weekday_traffic.xlsx')

# 독립 변수와 종속 변수 설정
X = df.loc[:, '0시':'23시']
y = df['혼잡']

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN 분류 모델 생성 및 학습
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 테스트 데이터 예측
y_pred = knn.predict(X_test_scaled)

# 혼동 행렬 출력
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n혼동 행렬:")
print(conf_matrix)

# 분류 성능 평가 지표 출력
prediction_report = classification_report(y_test, y_pred)
print("\n분류 성능 평가:")
print(prediction_report)

# 정확도 출력
accuracy = accuracy_score(y_test, y_pred)
print(f"\n모델의 정확도: {accuracy:.2f}")

# ----------------- 단항 및 다항 회귀 비교 -----------------

# 데이터 로드
weekdays_data = pd.read_excel('./data/weekday_traffic.xlsx')

# 독립 변수와 종속 변수 설정
days = np.arange(len(weekdays_data)).reshape(-1, 1)
traffic_at_8am = weekdays_data['8시'].values

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(days, traffic_at_8am, test_size=0.2, random_state=42)

# 단순 선형 회귀 모델 학습 및 예측
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# 단순 선형 회귀 RMSE 계산
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
print(f"단순 선형 회귀 RMSE: {rmse_linear:.2f}")

# 다항 회귀를 위한 특성 생성
poly = PolynomialFeatures(degree=2)
days_poly = poly.fit_transform(days)

# 데이터 분리
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(days_poly, traffic_at_8am, test_size=0.2, random_state=42)

# 다항 회귀 모델 학습 및 예측
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_poly)
y_pred_poly = poly_model.predict(X_test_poly)

# 다항 회귀 RMSE 계산
mse_poly = mean_squared_error(y_test_poly, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
print(f"다항 회귀 RMSE: {rmse_poly:.2f}")

# 마지막 날 교통량 예측 비교
predicted_traffic_linear = linear_model.predict(np.array([[len(weekdays_data)]]))[0]
last_day_index_poly = poly.transform(np.array([[len(weekdays_data)]]))
predicted_traffic_poly = poly_model.predict(last_day_index_poly)[0]
last_day_traffic = weekdays_data.iloc[-1]['8시']

# 결과 출력
print(f"\n실제 8시 교통량: {last_day_traffic:.2f}")
print(f"단순 선형 회귀 예측 교통량: {predicted_traffic_linear:.2f}")
print(f"다항 회귀 예측 교통량: {predicted_traffic_poly:.2f}")

# ----------------- 요일 기반 교통량 예측 -----------------

# 데이터 로드
file_path = './data/2023년_01월_서울시_교통량.xlsx'
data = pd.read_excel(file_path, sheet_name="2023년 01월")

# '일자' 컬럼을 datetime 형식으로 변환하고 요일 계산
data['일자'] = pd.to_datetime(data['일자'].astype(str), format='%Y%m%d')
data['요일'] = data['일자'].dt.weekday + 1

# 특정 지점 및 방향의 데이터 필터링
filtered_data = data[(data['지점명'] == '성산로(금화터널)') & (data['방향'] == '유입')]

# 피처와 타깃 값 설정
X = filtered_data[['요일']]
y = filtered_data['0시']

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 모델 평가
r2_score = model.score(X_test, y_test)
print(f"R²: {r2_score}")

# ----------------- 시간대별 교통량을 활용한 예측 -----------------

# 데이터 로드
file_path = './data/2023년_01월_서울시_교통량.xlsx'
data = pd.read_excel(file_path, sheet_name="2023년 01월")

# '일자' 컬럼을 datetime 형식으로 변환하고 요일 계산
data['일자'] = pd.to_datetime(data['일자'].astype(str))
data['요일'] = data['일자'].dt.weekday + 1

# 특정 지점 및 방향의 데이터 필터링
filtered_data = data[(data['지점명'] == '성산로(금화터널)') & (data['방향'] == '유입')]

# 피처와 타깃 값 설정 (요일과 1시~23시 교통량)
X = filtered_data[['요일']]
X = pd.concat([X, filtered_data[[f"{hour}시" for hour in range(1, 24)]]], axis=1)
y = filtered_data['0시']

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 모델 평가
r2_score = model.score(X_test, y_test)
print(f"R²: {r2_score}")
