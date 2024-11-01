import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split

# 1. 교통량 데이터 로드
weekdays_data = pd.read_excel('../data/weekday_traffic.xlsx') 

# 2. 독립 변수: 각 날을 표현하는 숫자 (1 ~ 43)
# 연속된 수의 배열을 생성하고 reshape()를 통해 차원을 변환 (참고: https://numpy.org/doc/stable/reference/generated/numpy.arange.html)
days = np.arange(1, 43).reshape(-1, 1)  # 각 날을 나타내는 인덱스를 독립 변수로 사용

# 3. 종속 변수: 각 날의 8시 교통량 데이터
traffic_at_8am = weekdays_data.loc[:, '8시'].values  # 열 이름이 '8시'인 교통량 데이터를 종속 변수로 사용

# 4. 데이터 분리 (훈련 데이터 80%, 테스트 데이터 20%)
# train_test_split으로 데이터를 훈련과 테스트 데이터로 분리(참고: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
# randomstate=42로 고정
X_train, X_test, y_train, y_test = train_test_split(days, traffic_at_8am, test_size=0.2, random_state=42)

# 5. 선형 회귀 모델 생성 및 학습 (훈련 데이터로 학습)
# LinearRegression 객체를 생성하여 fit() 함수로 학습 (참고: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
model = LinearRegression()
model.fit(X_train, y_train)

# 6. 마지막 날의 8시 교통량 예측
# 모델의 predict() 함수를 사용하여 예측값을 도출 (참고: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.predict)
predicted_traffic = model.predict(np.array([[43]]))

# 7. 마지막 날의 8시 실제 교통량 가져오기
# loc[]을 사용하여 마지막 날의 데이터를 가져옴 (참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html)
last_day_traffic = weekdays_data.loc[weekdays_data.index[-1], '8시']

# 8. 예측된 값과 실제 값을 출력
print(f"실제 Traffic: {last_day_traffic:.2f}, 예측한 Traffic: {predicted_traffic[0]:.2f}")
