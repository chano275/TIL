import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# 1. 교통량 데이터 로드
# 예시 파일 경로를 사용하여 엑셀 데이터를 불러옵니다.
weekdays_data = pd.read_excel('../data/weekday_traffic.xlsx')

# 2. 독립 변수: 각 날의 8시, 9시, 10시 교통량 데이터를 사용
# '8시', '9시', '10시' 열을 독립 변수로 선택하여 교통량 변수를 구성합니다.
X = weekdays_data.loc[:, ['8시', '9시', '10시']].values

# 3. 종속 변수: 혼잡 여부 (True/False -> 1/0 변환)
# '혼잡' 열을 종속 변수로 사용하며, 이 값이 True이면 1, False이면 0으로 변환합니다.
y = weekdays_data['혼잡'].astype(int).values

# 4. 데이터 분리 (훈련 데이터 80%, 테스트 데이터 20%)
# SVM 모델 학습을 위해 train_test_split을 사용해 데이터를 분리합니다.
# 참고: Scikit-learn - train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. 데이터 스케일링
# SVM 모델은 특성의 스케일에 민감하므로 StandardScaler를 사용해 데이터를 표준화합니다.
# 참고: Scikit-learn - StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 비선형 SVM 모델 생성 및 학습 (RBF 커널 사용)
# 비선형 분류 문제에 적합한 RBF 커널을 사용하여 SVM 모델을 생성하고 학습합니다.
# 참고: Scikit-learn - SVC (Support Vector Classifier): https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
model = SVC(kernel='rbf')  # RBF 커널을 사용한 SVM 모델
model.fit(X_train_scaled, y_train)

# 7. 테스트 데이터로 예측 수행
# 학습된 모델을 사용해 테스트 데이터를 예측합니다.
# 참고: Scikit-learn - predict 함수: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.predict
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