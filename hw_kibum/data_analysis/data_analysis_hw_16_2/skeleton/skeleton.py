import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 로드
# pandas의 read_excel을 사용하여 엑셀 파일에서 데이터를 불러옵니다.
# 참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html
weekdays_data = pd.read_excel('../data/weekday_traffic.xlsx')

# 2. 독립 변수와 종속 변수 설정
# '8시', '9시', '10시' 열을 독립 변수로 선택하여 교통량 변수를 구성합니다.
# '혼잡' 열을 종속 변수로 사용하며, 이 값이 True이면 1, False이면 0으로 변환합니다.
# astype(int)를 사용하여 True/False를 1/0으로 변환합니다.
# 참고: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
X = weekdays_data.loc[:, ['8시', '9시', '10시']].values
y = weekdays_data['혼잡'].astype(int).values

# 3. 데이터 분리
# train_test_split을 사용해 데이터를 학습 데이터와 테스트 데이터로 분리합니다.
# 테스트 데이터 비율은 20%로 설정합니다.
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. 모델 생성 및 학습
# DecisionTreeClassifier 모델을 생성하고 학습 데이터로 학습시킵니다.
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 5. 예측 수행
# 학습된 모델을 사용해 테스트 데이터에 대한 혼잡 여부를 예측합니다.
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.predict
y_pred = model.predict(X_test)

# 6. 성능 평가
# accuracy_score와 classification_report를 사용해 모델의 정확도 및 성능을 평가합니다.
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 7. 결과 출력
print(f"모델 정확도: {accuracy:.2f}")
print("분류 리포트:\n", report)
