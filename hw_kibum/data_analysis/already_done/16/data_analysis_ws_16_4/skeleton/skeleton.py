import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# 1. 교통량 데이터 로드
# Pandas를 사용하여 엑셀 파일을 읽어옵니다.
weekdays_data = pd.read_excel('../data/weekday_traffic.xlsx')

# 2. 독립 변수: 8시, 9시, 10시 교통량을 사용
# 독립 변수로 '8시', '9시', '10시' 열을 선택합니다.
X = weekdays_data.loc[:, ['8시', '9시', '10시']].values

# 3. 종속 변수: 혼잡 여부 (True/False -> 1/0 변환)
# '혼잡' 열을 사용하여 True/False 값을 1/0으로 변환합니다.
y = weekdays_data['혼잡'].astype(int).values

# 4. 데이터 분리 (훈련 데이터 80%, 테스트 데이터 20%)
# 데이터를 학습용과 테스트용으로 나눕니다.
# 참고: Scikit-learn - train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = _____________________________

# 5. 데이터 스케일링
# Logistic Regression 모델에서 성능을 향상시키기 위해 데이터를 표준화합니다.
# 참고: Scikit-learn - StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scaler = _____________________________
X_train_scaled = _____________________________
X_test_scaled = _____________________________

# 6. Logistic Regression 모델 생성 (L2 정규화 적용)
# Logistic Regression 모델을 생성하며, 기본적으로 L2 정규화(penalty='l2')가 적용됩니다.
# L2 정규화는 모델의 복잡도를 줄이고 과적합을 방지하기 위해 사용되며, 이를 통해 가중치 값들이 너무 커지는 것을 막습니다.
# C=1.0은 정규화 강도를 조정하는 하이퍼파라미터로, C 값이 클수록 정규화가 약해지고, 값이 작을수록 더 강한 정규화가 적용됩니다.
# 참고: Scikit-learn - LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
model = _____________________________  # L2 정규화
model.fit(X_train_scaled, y_train)

# 7. 테스트 데이터로 예측 수행
# 학습된 모델로 테스트 데이터를 예측합니다.
# 참고: Scikit-learn - predict 함수: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict
y_pred = _____________________________

# 8. 결과 평가
# 모델의 성능을 평가하기 위해 accuracy_score와 classification_report를 사용합니다.
# 참고: Scikit-learn - accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# 참고: Scikit-learn - classification_report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
accuracy = _____________________________
report = _____________________________

# 9. 결과 출력
print(f"모델 정확도: {accuracy:.2f}")
print("분류 리포트:\n", report)
