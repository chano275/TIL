import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. 데이터 로드
df = pd.read_excel('../data/weekday_traffic.xlsx')

# 2. 독립 변수와 종속 변수 설정
# 시간대별 교통량 데이터를 독립 변수(X)로, 혼잡 여부 데이터를 종속 변수(y)로 설정합니다.
X = df.loc[:, '0시':'23시']  # 시간대별 교통량 데이터를 독립 변수로 사용
y = df['혼잡']  # 혼잡 여부를 종속 변수로 설정

# 3. 학습 데이터와 테스트 데이터 분리
# train_test_split()을 사용하여 데이터를 훈련 세트(70%)와 테스트 세트(30%)로 분리합니다.
# stratify=y는 종속 변수(y)의 비율을 유지한 상태로 데이터를 나누기 위한 옵션입니다. (참고: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
X_train, X_test, y_train, y_test = __________________________________

# 4. 랜덤 포레스트 분류 모델 생성 및 학습
# RandomForestClassifier 객체를 생성하고 훈련 데이터를 사용하여 모델을 학습시킵니다. (참고: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
model = __________________________________
model.fit(X_train, y_train)

# 5. 테스트 셋에서 첫 번째 샘플 선택
# 첫 번째 샘플을 선택하여 일관된 결과를 보장합니다.
X_first = X_test.iloc[[0]]  # 테스트 셋의 첫 번째 샘플 선택

# 6. 해당 샘플의 날짜 가져오기
# df.loc[]을 사용하여 첫 번째 샘플의 날짜 데이터를 가져옵니다.
date_first = df.loc[X_test.index[0], '일자']  # 해당 샘플의 날짜

# 7. 첫 번째 샘플의 실제 혼잡 여부 가져오기
# y_test에서 첫 번째 샘플의 실제 혼잡 여부를 가져옵니다.
actual_congestion = y_test.iloc[0]

# 8. 예측 혼잡 여부 가져오기
# model.predict()를 사용하여 첫 번째 샘플의 혼잡 여부를 예측합니다. (참고: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict)
predicted_congestion = __________________________________

# 9. 결과 출력
# 첫 번째 샘플의 실제 혼잡 여부와 모델이 예측한 혼잡 여부를 출력하여 예측 성능을 확인합니다.
print(f"선택된 날짜: {date_first}")
print(f"실제 혼잡 여부: {'혼잡' if actual_congestion else '비혼잡'}")
print(f"예측 혼잡 여부: {'혼잡' if predicted_congestion else '비혼잡'}")
