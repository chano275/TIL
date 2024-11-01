import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. 데이터 로드
df = pd.read_excel('../data/weekday_traffic.xlsx')

# 2. 독립 변수(X)와 종속 변수(y) 설정
X = df.loc[:, '0시':'23시']  # 시간대별 교통량 데이터를 독립 변수로 사용
y = df['혼잡']  # 혼잡 여부를 종속 변수로 설정

# 3. 학습 데이터와 테스트 데이터 분리
# stratify=y를 추가하여 클래스 비율을 유지하도록 함
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

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
conf_matrix = __________________________________
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
prediction_report = __________________________________
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
accuracy = __________________________________
print(f"\n모델의 정확도(Accuracy): {accuracy:.2f}")
# 정확도(Accuracy):
# - 전체 데이터 중에서 모델이 올바르게 예측한 비율 ((TP + TN) / (전체 데이터 수))
#   => 정확도는 모델이 전반적으로 얼마나 잘 예측했는지를 나타내는 지표

