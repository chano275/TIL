import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
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
# '혼잡' 열을 사용하여 1/0으로 변환합니다.
y = weekdays_data['혼잡'].astype(int).values

# 4. 데이터 분리 (훈련 데이터 80%, 테스트 데이터 20%)
# 데이터를 학습용과 테스트용으로 나눕니다.
# 참고: Scikit-learn - train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. 데이터 스케일링
# 데이터를 표준화하여 모델 성능을 향상시킵니다.
# 참고: Scikit-learn - StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Gradient Boosting 모델 생성 및 학습
'''
Gradient Boosting은 여러 개의 약한 학습기(주로 결정 트리)를 순차적으로 학습하여 예측 성능을 향상시키는 앙상블 학습 방법 중 하나입니다. 
각 단계에서 이전 모델이 잘못 예측한 데이터를 더 잘 맞추도록 가중치를 부여하여 새로운 모델을 학습시킵니다. 
최종적으로 모든 약한 학습기의 예측 결과를 합산하여 강력한 예측 모델을 생성합니다.

GradientBoostingClassifier는 이 아이디어를 기반으로 한 분류 모델입니다. 
모델은 먼저 예측이 틀린 데이터에 더 많은 비중을 두고 학습하며, 이를 반복해 나가면서 오차를 줄입니다. 
이 과정은 손실 함수의 기울기를 최소화하는 방식으로 진행되기 때문에 'Gradient'라는 용어가 사용됩니다. 
일반적으로, 과적합을 방지하기 위해 트리의 깊이나 학습률(learning rate) 등을 조절할 수 있습니다.
'''
# GradientBoostingClassifier를 사용하여 모델을 생성하고 학습합니다.
# n_estimators=10, learning_rate=0.1, max_depth=3, random_state=42를 설정해 맞춰줍니다.
# 참고: Scikit-learn - GradientBoostingClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=3)  # 모델 개수
model.fit(X_train_scaled, y_train)

# 7. 테스트 데이터로 예측 수행
# 학습된 모델로 테스트 데이터를 예측합니다.
# 참고: Scikit-learn - predict 함수: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier.predict
y_pred = model.predict(X_test_scaled)

# 8. 결과 평가
# 모델의 성능을 평가하기 위해 accuracy_score와 classification_report를 사용합니다.
# 참고: Scikit-learn - accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# 참고: Scikit-learn - classification_report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 9. 결과 출력
print(f"모델 정확도: {accuracy:.2f}")
print("분류 리포트:\n", report)
