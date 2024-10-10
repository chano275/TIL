import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. 데이터 로드
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
X_train, X_test, y_train, y_test = __________________________________

# 5. 선형 회귀 모델 생성 및 학습 (단항 회귀)
# LinearRegression 객체를 생성하고, 훈련 데이터를 학습시킵니다. (참고: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
linear_model = __________________________________
linear_model.fit(X_train, y_train)

# 6. 테스트 데이터로 예측
# 학습된 모델을 사용하여 테스트 데이터를 예측합니다.
y_pred_linear = __________________________________

# 7. 성능 평가 (MSE, RMSE)
# mean_squared_error()를 사용하여 MSE(평균 제곱 오차)를 계산하고, 이를 바탕으로 RMSE를 계산합니다. (참고: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
mse_linear = __________________________________
rmse_linear = __________________________________
print(f"단항 회귀 모델의 RMSE: {rmse_linear:.2f}")

# ----------------- 다항 회귀 -----------------

# 8. 다항 특성 생성 (2차 다항식 사용)
# PolynomialFeatures()를 사용해 2차 다항식을 생성합니다. (참고: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
poly = __________________________________
# 독립 변수를 다항식으로 변환하는 과정입니다.
days_poly = poly.fit_transform(days)

# 9. 데이터 분리 (훈련 데이터 80%, 테스트 데이터 20%)
# 다항 회귀에서도 동일하게 데이터를 분리합니다.
X_train_poly, X_test_poly, y_train_poly, y_test_poly = __________________________________

# 10. 선형 회귀 모델 생성 및 학습 (다항 회귀)
# 다항 회귀에서도 선형 회귀 모델을 학습시킵니다.
poly_model = __________________________________
poly_model.fit(X_train_poly, y_train_poly)

# 11. 테스트 데이터로 예측
# 학습된 다항 회귀 모델을 사용하여 테스트 데이터를 예측합니다.
y_pred_poly = __________________________________

# 12. 성능 평가 (MSE, RMSE)
# 다항 회귀의 성능을 MSE, RMSE로 평가합니다. (참고: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
mse_poly = __________________________________
rmse_poly = __________________________________
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