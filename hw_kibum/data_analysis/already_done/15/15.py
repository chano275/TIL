import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error


weekdays_data = pd.read_excel('../data/weekday_traffic.xlsx') 

# 독립 변수: 각 날의 8시만 사용 (날짜별로 동일한 시간) - # np.arange()는 연속된 수의 배열을 생성하며, reshape()은 차원을 변환 (참고: https://numpy.org/doc/stable/reference/generated/numpy.arange.html)
days = np.arange(len(weekdays_data)).reshape(-1, 1)  # 각 날을 나타내는 인덱스를 독립 변수로 사용

# 종속 변수: 각 날의 8시 교통량 데이터
traffic_at_8am = weekdays_data.loc[:, '8시'].values  # 열 이름이 '8시'인 교통량 데이터를 종속 변수로 사용

# 데이터 분리 (훈련 데이터 80%, 테스트 데이터 20%) - # train_test_split은 데이터를 훈련과 테스트 데이터로 분리하는 함수 (참고: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) /  randomstate=42로 고정
# 선형 회귀 모델 생성 및 학습 (훈련 데이터로 학습) - LinearRegression 객체를 생성하여 fit() 함수로 학습 (참고: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
X_train, X_test, y_train, y_test = train_test_split(days, traffic_at_8am, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 마지막 날의 8시 교통량 예측 - 모델의 predict() 함수를 사용하여 예측값을 도출 (참고: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.predict)
predicted_traffic = model.predict(np.array([[len(weekdays_data)]]))

# 마지막 날의 8시 실제 교통량 가져오기 - loc[]을 사용하여 마지막 날의 데이터를 가져옴 (참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html)
last_day_traffic = weekdays_data.loc[weekdays_data.index[-1], '8시']

# 예측된 값과 실제 값을 출력
print(f"실제 Traffic: {last_day_traffic:.2f}, 예측한 Traffic: {predicted_traffic[0]:.2f}")

#####

df = pd.read_excel('../data/weekday_traffic.xlsx')

# 독립 변수와 종속 변수 설정 - 시간대별 교통량 데이터를 독립 변수(X)로, 혼잡 여부 데이터를 종속 변수(y)로 설정합니다.
X = df.loc[:, '0시':'23시'] 
y = df['혼잡']

# 학습 데이터와 테스트 데이터 분리 - train_test_split()을 사용하여 데이터를 훈련 세트(70%)와 테스트 세트(30%)로 분리합니다.
# stratify=y는 종속 변수(y)의 비율을 유지한 상태로 데이터를 나누기 위한 옵션 (참고: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 랜덤 포레스트 분류 모델 생성 및 학습 - RandomForestClassifier 객체를 생성하고 훈련 데이터를 사용 > 모델 학습 (참고: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 테스트 셋에서 첫 번째 샘플 선택 - 첫 번째 샘플을 선택하여 일관된 결과를 보장
X_first = X_test.iloc[[0]]  # 테스트 셋의 첫 번째 샘플 선택

# 해당 샘플의 날짜 가져오기 - df.loc[]을 사용하여 첫 번째 샘플의 날짜 데이터를 가져옵니다.
date_first = df.loc[X_test.index[0], '일자']  # 해당 샘플의 날짜

# 첫 번째 샘플의 실제 혼잡 여부 가져오기 - y_test에서 첫 번째 샘플의 실제 혼잡 여부를 가져옵니다.
actual_congestion = y_test.iloc[0]

# 예측 혼잡 여부 가져오기 - model.predict()를 사용하여 첫 번째 샘플의 혼잡 여부를 예측 (참고: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict)
predicted_congestion = model.predict(X_first)[0]

# 결과 출력 - 첫 번째 샘플의 실제 혼잡 여부와 모델이 예측한 혼잡 여부를 출력하여 예측 성능을 확인합니다. 
print(f"선택된 날짜: {date_first}")
print(f"실제 혼잡 여부: {'혼잡' if actual_congestion else '비혼잡'}")
print(f"예측 혼잡 여부: {'혼잡' if predicted_congestion else '비혼잡'}")

#####

df = pd.read_excel('../data/weekday_traffic.xlsx')

# 독립 변수(X)와 종속 변수(y) 설정 - 시간대별 교통량 데이터를 독립 변수(X)로, 혼잡 여부 데이터를 종속 변수(y)로 설정합니다.
X = df.loc[:, '0시':'23시']  # 시간대별 교통량 데이터를 독립 변수로 설정
y = df['혼잡']  # 혼잡 여부를 종속 변수로 설정

# 학습 데이터와 테스트 데이터 분리 - train_test_split()을 사용해 데이터를 훈련 세트(70%)와 테스트 세트(30%)로 분리 / stratify=y 옵션은 종속 변수(y)의 클래스 비율을 유지하도록 합니다. (참고: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
 
# 데이터 스케일링 (표준화) - StandardScaler()를 사용해 데이터의 평균을 0, 표준편차를 1로 맞추어 표준화
# 훈련 데이터에서 스케일링 기준을 학습하고, 학습된 기준으로 훈련 및 테스트 데이터를 변환 (참고: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
scaler = StandardScaler()  # scaler를 사용해 변환

# fit_transform(): 훈련 데이터에 대해 평균과 표준편차를 계산하고, 그 값으로 데이터를 표준화합니다.
# transform(): 훈련 데이터에서 계산된 평균과 표준편차를 사용하여 테스트 데이터를 표준화합니다.
X_train_scaled = scaler.fit_transform(X_train)  # 훈련 데이터를 스케일링
X_test_scaled = scaler.transform(X_test)  # 테스트 데이터를 스케일링

# KNN 분류 모델 생성 및 학습 - KNeighborsClassifier()를 사용하여 KNN 분류 모델을 생성하고, 훈련 데이터를 이용해 학습시킴
# n_neighbors=5로 설정해, K값을 5로 사용하여 가장 가까운 5개의 이웃을 기준으로 예측 (참고: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 테스트 셋에서 첫 번째 샘플 선택 및 결과 출력 - 무작위 선택 대신, 첫 번째 샘플을 선택하여 항상 일관된 결과를 보장
selected_sample = X_test.iloc[0:1]  # 테스트 세트에서 첫 번째 샘플을 선택
selected_sample_scaled = X_test_scaled[0].reshape(1, -1)  # 선택한 샘플을 스케일링된 값으로 변환

# 해당 샘플의 실제 혼잡 여부 - y_test에서 첫 번째 샘플의 실제 혼잡 여부를 가져옴
actual_congestion = y_test.iloc[0]

# 모델을 사용해 예측된 혼잡 여부 - KNN 모델을 사용해 첫 번째 샘플의 혼잡 여부를 예측 (참고: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
predicted_congestion = knn.predict(selected_sample_scaled)[0]

# 해당 샘플의 날짜 추출 - df에서 선택된 샘플의 날짜 정보를 가져옵니다.
selected_date = df.iloc[X_test.index[0]]['일자']

# 10. 결과 출력
# 첫 번째 샘플의 실제 혼잡 여부와 예측된 혼잡 여부를 비교하여 출력합니다.
print(f"선택된 날짜: {selected_date}")
print(f"실제 혼잡 여부: {'혼잡' if actual_congestion else '비혼잡'}")
print(f"예측 혼잡 여부: {'혼잡' if predicted_congestion else '비혼잡'}")

#####

df = pd.read_excel('../data/weekday_traffic.xlsx')

# 2. 독립 변수(X)와 종속 변수(y) 설정
X = df.loc[:, '0시':'23시']  # 시간대별 교통량 데이터를 독립 변수로 사용
y = df['혼잡']  # 혼잡 여부를 종속 변수로 설정

# 3. 학습 데이터와 테스트 데이터 분리 - stratify=y를 추가하여 클래스 비율을 유지하도록 함
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 4. 데이터 스케일링 (표준화)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. KNN 분류 모델 생성 및 학습
knn = KNeighborsClassifier(n_neighbors=5)  # K 값을 5로 설정
knn.fit(X_train_scaled, y_train)

# 6. 테스트 데이터에 대한 예측
y_pred = knn.predict(X_test_scaled)

# 7. 모델 성능 평가
# 혼동 행렬 출력
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n혼동 행렬(Confusion Matrix):")
print(conf_matrix)
# 혼동 행렬(Confusion Matrix):
# - TP (True Positive): 실제로 True인 데이터 중에서 모델이 True라고 예측한 것
# - TN (True Negative): 실제로 False인 데이터 중에서 모델이 False라고 예측한 것
# - FP (False Positive): 실제로 False인 데이터 중에서 모델이 True라고 잘못 예측한 것 (False Positive)
# - FN (False Negative): 실제로 True인 데이터 중에서 모델이 False라고 잘못 예측한 것 (False Negative)
# - 행: 실제 값 (True Labels)
# - 열: 예측 값 (Predicted Labels)
# [[0, 1] 위치는 실제로 비혼잡(0)인데 혼잡(1)으로 잘못 예측된 경우를 의미
#  [1, 0]] 위치는 실제로 혼잡(1)인데 비혼잡(0)으로 잘못 예측된 경우를 의미

# 분류 성능 평가 지표 출력 (정확도, 정밀도, 재현율, F1 점수 등)
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
prediction_report = classification_report(y_test, y_pred)
print("\n분류 성능 평가(Classification Report):")
print(prediction_report)
# 분류 성능 평가(Classification Report):
# - Precision (정밀도): 모델이 True라고 예측한 것 중 실제로 True인 데이터의 비율 (TP / (TP + FP))
#   => False Positive를 줄이는 데 중점을 둔 지표
# - Recall (재현율): 실제로 True인 것 중에서 모델이 True라고 올바르게 예측한 비율 (TP / (TP + FN))
#   => False Negative를 줄이는 데 중점을 둔 지표
# - F1 Score: 정밀도와 재현율의 조화 평균 (2 * (Precision * Recall) / (Precision + Recall))
#   => Precision과 Recall 간의 균형을 중요시할 때 사용
# - Support: 각 클래스에 속하는 실제 데이터의 수

# 정확도 출력
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
accuracy = accuracy_score(y_test, y_pred)
print(f"\n모델의 정확도(Accuracy): {accuracy:.2f}")
# 정확도(Accuracy):
# - 전체 데이터 중에서 모델이 올바르게 예측한 비율 ((TP + TN) / (전체 데이터 수))
#   => 정확도는 모델이 전반적으로 얼마나 잘 예측했는지를 나타내는 지표

#####

weekdays_data = pd.read_excel('../data/weekday_traffic.xlsx')

# 2. 독립 변수: 각 날의 8시만 사용 (날짜별로 동일한 시간)
# numpy의 arange() 함수와 reshape()을 사용하여 각 날을 나타내는 인덱스 배열을 생성합니다. (참고: https://numpy.org/doc/stable/reference/generated/numpy.arange.html)
days = np.arange(len(weekdays_data)).reshape(-1, 1)

# 3. 종속 변수: 각 날의 8시 교통량 데이터
# 교통량 데이터는 '8시'라는 컬럼명을 기준으로 가져옵니다. (참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html)
traffic_at_8am = weekdays_data['8시'].values

# ----------------- 단항 회귀 (선형 회귀) -----------------

# 4. 데이터 분리 (훈련 데이터 80%, 테스트 데이터 20%)
# train_test_split()을 사용해 데이터를 훈련 세트와 테스트 세트로 나눕니다. (참고: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
X_train, X_test, y_train, y_test = train_test_split(days, traffic_at_8am, test_size=0.2, random_state=42)

# 5. 선형 회귀 모델 생성 및 학습 (단항 회귀)
# LinearRegression 객체를 생성하고, 훈련 데이터를 학습시킵니다. (참고: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 6. 테스트 데이터로 예측
# 학습된 모델을 사용하여 테스트 데이터를 예측합니다.
y_pred_linear = linear_model.predict(X_test)

# 7. 성능 평가 (MSE, RMSE)
# mean_squared_error()를 사용하여 MSE(평균 제곱 오차)를 계산하고, 이를 바탕으로 RMSE를 계산합니다. (참고: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
print(f"단항 회귀 모델의 RMSE: {rmse_linear:.2f}")

# ----------------- 다항 회귀 -----------------

# 8. 다항 특성 생성 (2차 다항식 사용)
# PolynomialFeatures()를 사용해 2차 다항식을 생성합니다. (참고: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
poly = PolynomialFeatures(degree=2)
days_poly = poly.fit_transform(days)

# 9. 데이터 분리 (훈련 데이터 80%, 테스트 데이터 20%)
# 다항 회귀에서도 동일하게 데이터를 분리합니다.
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(days_poly, traffic_at_8am, test_size=0.2, random_state=42)

# 10. 선형 회귀 모델 생성 및 학습 (다항 회귀)
# 다항 회귀에서도 선형 회귀 모델을 학습시킵니다.
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_poly)

# 11. 테스트 데이터로 예측
# 학습된 다항 회귀 모델을 사용하여 테스트 데이터를 예측합니다.
y_pred_poly = poly_model.predict(X_test_poly)

# 12. 성능 평가 (MSE, RMSE)
# 다항 회귀의 성능을 MSE, RMSE로 평가합니다. (참고: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
mse_poly = mean_squared_error(y_test_poly, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
print(f"다항 회귀 모델의 RMSE: {rmse_poly:.2f}")

# ----------------- 마지막 날 교통량 예측 비교 -----------------

# 13. 마지막 날의 교통량 예측 (단항 회귀)
# 학습된 선형 회귀 모델로 마지막 날의 8시 교통량을 예측합니다.
predicted_traffic_linear = linear_model.predict(np.array([[len(weekdays_data)]]))[0]

# 14. 마지막 날의 교통량 예측 (다항 회귀)
# 학습된 다항 회귀 모델로 마지막 날의 8시 교통량을 예측합니다.
last_day_index_poly = poly.transform(np.array([[len(weekdays_data)]]))
predicted_traffic_poly = poly_model.predict(last_day_index_poly)[0]

# 15. 마지막 날의 실제 교통량과 예측 교통량 비교
# 실제 교통량과 예측된 값을 출력하여 모델 성능을 비교합니다.
last_day_traffic = weekdays_data.iloc[-1]['8시']
print(f"\n실제 8시 교통량: {last_day_traffic:.2f}")
print(f"단항 회귀로 예측한 8시 교통량: {predicted_traffic_linear:.2f}")
print(f"다항 회귀로 예측한 8시 교통량: {predicted_traffic_poly:.2f}")

##### 

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. 데이터 불러오기
# pandas의 read_excel을 사용하여 엑셀 파일에서 데이터를 불러옵니다.
# file_path는 불러올 엑셀 파일의 경로입니다.
# sheet_name은 불러올 시트의 이름을 지정하는데, 여기서는 '2023년 01월' 데이터를 가져옵니다.
# 참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html
file_path = '../data/2023년 01월 서울시 교통량.xlsx'
data = pd.read_excel(file_path, sheet_name="2023년 01월")

# 데이터를 살펴보면 현재 일자와 요일이 일치하지 않습니다. 그렇기 때문에 일자 칼럼을 이용하여 요일을 다시 계산해야 합니다.

# 2. '일자' 컬럼을 datetime 형식으로 변환
# 이를 datetime 형식으로 변환하면 날짜 관련 작업이 더 쉬워집니다.
# pandas의 to_datetime 함수를 사용하여 '일자' 데이터를 날짜 형식으로 변환합니다.
# (참고: '일자' 컬럼은 날짜를 나타내는 값인데, 이 컬럼의 각 값들은 정수 형식(YYYYMMDD)으로 되어 있습니다.)
# 참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html
data['일자'] = pd.to_datetime(data['일자'].astype(str), format='%Y%m%d')

# 3. 날짜를 기반으로 요일을 자동으로 계산
# 요일이 정수로 변환되면 학습 모델에서 이를 숫자형 데이터로 처리할 수 있습니다.
# datetime 형식으로 변환된 '일자' 컬럼을 기반으로 요일을 계산합니다.
# dt.weekday는 '월요일'=0, '일요일'=6을 반환하므로, +1을 해주어 '월요일'=1, '일요일'=7로 변환합니다.
# 참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.html
data['요일'] = data['일자'].dt.weekday + 1

# 4. 데이터 필터링
# '성산로(금화터널)' 지점에서 '유입' 방향으로 들어오는 데이터만을 선택합니다.
# 데이터를 먼저 필터링하여 분석에 필요한 데이터만 남깁니다.
# Boolean indexing을 사용하여 조건에 맞는 데이터만 필터링할 수 있습니다.
# 예: filtered_data = 데이터프레임[데이터프레임['컬럼명'] == '값' & 데이터프레임['컬럼명'] == '값']
# 참고: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#boolean-indexing
filtered_data = data[(data['지점명'] == '성산로(금화터널)') & (data['방향'] == '유입')]

# 5. 피처와 타깃 값 설정
# '요일'을 독립 변수로, '0시' 교통량을 종속 변수로 설정합니다.
# 참고: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
X = filtered_data[['요일']]
y = filtered_data['0시']

# 6. 학습 데이터와 테스트 데이터 분할
# 학습 데이터와 테스트 데이터를 나누는 함수로 train_test_split을 사용합니다.
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 7. 선형 회귀 모델 정의 및 학습
# scikit-learn의 LinearRegression 모델을 사용하여 학습 데이터를 학습시킵니다.
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
model = LinearRegression()
model.fit(X_train, y_train)

# 8. 모델 평가
# 학습된 모델을 평가하기 위해 R² 점수를 사용합니다. 1에 가까울수록 좋은 성능을 의미합니다.
# scikit-learn의 score() 함수는 기본적으로 R² 점수를 반환합니다.
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.score
r2_score = model.score(X_test, y_test)
print(f"R²: {r2_score}")

#####

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. 데이터 불러오기
# pandas의 read_excel을 사용하여 엑셀 파일에서 데이터를 불러옵니다.
# file_path는 불러올 엑셀 파일의 경로입니다.
# sheet_name은 불러올 시트의 이름을 지정합니다. 여기서는 '2023년 01월' 데이터를 가져옵니다.
# 참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html
file_path = '../data/2023년 01월 서울시 교통량.xlsx'
data = pd.read_excel(file_path, sheet_name="2023년 01월")

# 데이터를 살펴보면 현재 일자와 요일이 일치하지 않습니다. 그렇기 때문에 일자 칼럼을 이용하여 요일을 다시 계산해야 합니다.

# 2. '일자' 컬럼을 datetime 형식으로 변환
# 이를 datetime 형식으로 변환하면 날짜 관련 작업이 더 쉬워집니다.
# pandas의 to_datetime 함수를 사용하여 '일자' 데이터를 날짜 형식으로 변환합니다.
# (참고: '일자' 컬럼은 날짜를 나타내는 값인데, 이 컬럼의 각 값들은 정수 형식(YYYYMMDD)으로 되어 있습니다.)
# 참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html
data['일자'] = pd.to_datetime(data['일자'].astype(str))

# 3. 날짜를 기반으로 요일을 자동으로 계산
# 요일 정보는 모델에서 숫자형 데이터로 처리할 수 있도록 변환되어야 합니다.
# datetime 형식으로 변환된 '일자' 컬럼을 기반으로 요일을 계산합니다.
# dt.weekday는 '월요일'=0, '일요일'=6을 반환하므로, +1을 해주어 '월요일'=1, '일요일'=7로 변환합니다.
# 참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.html
data['요일'] = data['일자'].dt.weekday + 1

# 4. 시간대 변수를 그대로 사용
# 기존 데이터셋에서 '0시'부터 '23시'까지 시간대별 교통량 데이터가 이미 존재합니다.
# 이를 독립 변수로 사용하여 각 시간대의 교통량 정보를 피처로 추가합니다.
# 시간대 컬럼은 '0시', '1시'... '23시' 등의 이름으로 존재하므로 그대로 사용할 수 있습니다.

# 5. 데이터 필터링
# '성산로(금화터널)' 지점에서 '유입' 방향으로 들어오는 데이터만 필터링합니다.
# Boolean indexing을 사용하여 조건을 만족하는 행만 선택합니다.
# Boolean indexing: 조건이 True인 데이터만 선택합니다.
# 예: filtered_data = 데이터프레임[(데이터프레임['컬럼명'] == '값') & (데이터프레임['컬럼명'] == '값')]
# 참고: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#boolean-indexing
filtered_data = data[(data['지점명'] == '성산로(금화터널)') & (data['방향'] == '유입')]

# 6. 피처와 타깃 값 설정
# '요일'과 '시간대' (1시~23시)의 정보를 피처로 사용하고, '0시' 시간대의 교통량을 타깃 값으로 설정합니다.
# 시간대 정보는 이미 데이터셋에서 '1시', '2시' 등의 컬럼으로 존재하므로 이를 독립 변수로 추가합니다.
# pandas의 concat 함수를 사용하여 피처를 결합합니다.
# 참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
# 힌트: 시간대 컬럼은 '1시', '2시', ..., '23시'로 되어 있으며, 이를 for 루프를 통해 추가할 수 있습니다.
# 시간대 컬럼의 이름은 문자열로 처리됩니다. 이를 이용해 여러 열을 동시에 추가할 수 있습니다.
# 예: str(hour) + '시'를 사용하여 1시, 2시 등의 컬럼명을 만들어 보세요.
X = filtered_data[['요일']]  # 요일 정보를 독립 변수로 사용
X = pd.concat([X, filtered_data[[str(hour) + '시' for hour in range(1, 24)]]], axis=1)  # 시간대별 교통량 추가
y = filtered_data['0시']  # 타깃 값으로 0시의 교통량 설정

# 7. 학습 데이터와 테스트 데이터 분할
# 학습 데이터와 테스트 데이터를 나누는 함수로 train_test_split을 사용합니다.
# test_size=0.2는 데이터의 20%를 테스트 데이터로 사용한다는 의미입니다.
# random_state=42는 무작위 분할을 고정하여 재현성을 보장합니다.
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. 선형 회귀 모델 정의 및 학습
# scikit-learn의 LinearRegression 모델을 사용하여 학습 데이터를 학습시킵니다.
# 모델을 정의한 후, fit 함수를 사용하여 학습 데이터를 학습합니다.
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
model = LinearRegression()
model.fit(X_train, y_train)

# 9. 모델 평가
# 학습된 모델을 평가하기 위해 R² 점수를 사용합니다. 1에 가까울수록 성능이 좋은 모델입니다.
# score() 함수는 기본적으로 R² 점수를 반환합니다.
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.score
r2_score = model.score(X_test, y_test)
print(f"R²: {r2_score}")
