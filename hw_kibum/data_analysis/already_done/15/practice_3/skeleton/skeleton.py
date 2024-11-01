import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1. 데이터 로드
# pandas.read_excel()을 사용해 엑셀 파일에서 교통량 데이터를 불러옵니다. (참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html)
df = pd.read_excel('../data/weekday_traffic.xlsx')

# 2. 독립 변수(X)와 종속 변수(y) 설정
# 시간대별 교통량 데이터를 독립 변수(X)로, 혼잡 여부 데이터를 종속 변수(y)로 설정합니다.
X = df.loc[:, '0시':'23시']  # 시간대별 교통량 데이터를 독립 변수로 설정
y = df['혼잡']  # 혼잡 여부를 종속 변수로 설정

# 3. 학습 데이터와 테스트 데이터 분리
# train_test_split()을 사용해 데이터를 훈련 세트(70%)와 테스트 세트(30%)로 분리합니다.
# stratify=y 옵션은 종속 변수(y)의 클래스 비율을 유지하도록 합니다. (참고: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# 4. 데이터 스케일링 (표준화)
# StandardScaler()를 사용해 데이터의 평균을 0, 표준편차를 1로 맞추어 표준화합니다.
# 훈련 데이터에서 스케일링 기준을 학습하고, 학습된 기준으로 훈련 및 테스트 데이터를 변환합니다. (참고: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
scaler = StandardScaler()
# scaler를 사용해 변환합니다.
# fit_transform(): 훈련 데이터에 대해 평균과 표준편차를 계산하고, 그 값으로 데이터를 표준화합니다.
# transform(): 훈련 데이터에서 계산된 평균과 표준편차를 사용하여 테스트 데이터를 표준화합니다.
X_train_scaled = scaler.fit_transform(X_train)  # 훈련 데이터를 스케일링
X_test_scaled = scaler.fit(X_test)  # 테스트 데이터를 스케일링

# 5. KNN 분류 모델 생성 및 학습
# KNeighborsClassifier()를 사용하여 KNN 분류 모델을 생성하고, 훈련 데이터를 이용해 학습시킵니다.
# n_neighbors=5로 설정해, K값을 5로 사용하여 가장 가까운 5개의 이웃을 기준으로 예측합니다. (참고: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 6. 테스트 셋에서 첫 번째 샘플 선택 및 결과 출력
# 무작위 선택 대신, 첫 번째 샘플을 선택하여 항상 일관된 결과를 보장합니다.
selected_sample = X_test.iloc[0:1]  # 테스트 세트에서 첫 번째 샘플을 선택
selected_sample_scaled = X_test_scaled[0].reshape(1, -1)  # 선택한 샘플을 스케일링된 값으로 변환

# 7. 해당 샘플의 실제 혼잡 여부
# y_test에서 첫 번째 샘플의 실제 혼잡 여부를 가져옵니다.
actual_congestion = y_test.iloc[0]

# 8. 모델을 사용해 예측된 혼잡 여부
# KNN 모델을 사용해 첫 번째 샘플의 혼잡 여부를 예측합니다. (참고: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
predicted_congestion = knn(selected_sample)

# 9. 해당 샘플의 날짜 추출
# df에서 선택된 샘플의 날짜 정보를 가져옵니다.
selected_date = df.iloc[X_test.index[0]]['일자']

# 10. 결과 출력
# 첫 번째 샘플의 실제 혼잡 여부와 예측된 혼잡 여부를 비교하여 출력합니다.
print(f"선택된 날짜: {selected_date}")
print(f"실제 혼잡 여부: {'혼잡' if actual_congestion else '비혼잡'}")
print(f"예측 혼잡 여부: {'혼잡' if predicted_congestion else '비혼잡'}")
