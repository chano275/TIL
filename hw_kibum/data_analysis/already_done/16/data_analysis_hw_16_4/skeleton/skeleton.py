from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# 1. 데이터 로드
weekdays_data = pd.read_excel('../data/weekday_traffic.xlsx')

# 2. 독립 변수와 종속 변수 설정
# '8시', '9시', '10시' 열을 독립 변수로 선택하여 교통량 변수를 구성합니다.
#  종속 변수: 혼잡 여부 (True/False -> 1/0 변환)
# '혼잡' 열을 종속 변수로 사용하며, 이 값이 True이면 1, False이면 0으로 변환합니다.
X = weekdays_data.loc[:, ['8시', '9시', '10시']].values
y = weekdays_data['혼잡'].astype(int).values

# 3. 데이터 분리 (훈련 데이터 80%, 테스트 데이터 20%)
# 데이터를 학습용과 테스트용으로 나눕니다.
# 참고: Scikit-learn - train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. RandomForest 모델 생성 및 학습
# 랜덤 포레스트 - 배깅 방식
# 배깅(bagging) 방식의 랜덤 포레스트 모델을 생성합니다.
# - n_estimators=10: 트리의 개수를 10개로 설정
# - random_state=42: 결과 재현을 위해 고정된 난수 시드를 설정
# 참고: Scikit-learn - RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# 5. 예측 수행
# 학습된 모델을 사용해 테스트 데이터에 대한 혼잡 여부를 예측합니다.
# 참고: Scikit-learn - predict 함수: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.predict
y_pred = model.predict(X_test)

# 6. 성능 평가
# accuracy_score와 classification_report를 사용해 모델의 정확도 및 성능을 평가합니다.
# 참고: Scikit-learn - accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# 참고: Scikit-learn - classification_report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 7. 결과 출력
print(f"모델 정확도: {accuracy:.2f}")
print("분류 리포트:\n", report)
