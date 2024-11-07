import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


weekdays_data = pd.read_excel('../data/weekday_traffic.xlsx')

# 독립 변수: 각 날의 8시, 9시, 10시 교통량 데이터를 사용 - '8시', '9시', '10시' 열을 독립 변수로 선택하여 교통량 변수를 구성합니다.
# 종속 변수: 혼잡 여부 (True/False -> 1/0 변환) - '혼잡' 열을 종속 변수로 사용하며, 이 값이 True이면 1, False이면 0으로 변환합니다.
X = weekdays_data.loc[:, ['8시', '9시', '10시']].values
y = weekdays_data['혼잡'].astype(int).values

# 데이터 분리 (훈련 데이터 80%, 테스트 데이터 20%) - SVM 모델 학습을 위해 train_test_split을 사용해 데이터를 분리합니다. - 참고: Scikit-learn - train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링 (SVM 성능 향상을 위해 표준화) - SVM 모델은 특성의 스케일에 민감하므로 StandardScaler를 사용해 데이터를 표준화 - 참고: Scikit-learn - StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

########################### 모델 종류에 따라 여기만 다름 ########################################################
### SVM 모델 생성 및 학습 (훈련 데이터로 학습) - SVM 모델을 생성하여 학습합니다. 여기서는 선형 커널을 사용합니다. - 참고: Scikit-learn - SVC: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

### cf> RBF 커널 
# # 6. 비선형 SVM 모델 생성 및 학습 (RBF 커널 사용) - 비선형 분류 문제에 적합한 RBF 커널을 사용하여 SVM 모델을 생성하고 학습합니다. - 참고: Scikit-learn - SVC (Support Vector Classifier): https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# model = SVC(kernel='rbf')  # RBF 커널을 사용한 SVM 모델

### cf> Gradient Boosting 모델 생성 및 학습 - # GradientBoostingClassifier를 사용하여 모델을 생성하고 학습
# n_estimators=10, learning_rate=0.1, max_depth=3, random_state=42를 설정해 맞춰줍니다. - 참고: Scikit-learn - GradientBoostingClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
'''
Gradient Boosting은 여러 개의 약한 학습기(주로 결정 트리)를 순차적으로 학습하여 예측 성능을 향상시키는 앙상블 학습 방법 중 하나
각 단계에서 이전 모델이 잘못 예측한 데이터를 더 잘 맞추도록 가중치를 부여하여 새로운 모델을 학습
최종적으로 모든 약한 학습기의 예측 결과를 합산하여 강력한 예측 모델을 생성합니다.

GradientBoostingClassifier는 이 아이디어를 기반으로 한 분류 모델입니다. 
모델은 먼저 예측이 틀린 데이터에 더 많은 비중을 두고 학습하며, 이를 반복해 나가면서 오차를 줄입니다. 
이 과정은 손실 함수의 기울기를 최소화하는 방식으로 진행되기 때문에 'Gradient'라는 용어가 사용됩니다. 
일반적으로, 과적합을 방지하기 위해 트리의 깊이나 학습률(learning rate) 등을 조절할 수 있습니다.
'''
# model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=3, random_state=42)
# model.fit(X_train_scaled, y_train)


### cf> Logistic Regression 모델 생성 (L2 정규화 적용) - Logistic Regression 모델을 생성하며, 기본적으로 L2 정규화(penalty='l2')가 적용
# L2 정규화는 모델의 복잡도를 줄이고 과적합을 방지하기 위해 사용되며, 이를 통해 가중치 값들이 너무 커지는 것을 막습니다.
# C=1.0은 정규화 강도를 조정하는 하이퍼파라미터로, C 값이 클수록 정규화가 약해지고, 값이 작을수록 더 강한 정규화가 적용됩니다.
# 참고: Scikit-learn - LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# model = LogisticRegression(penalty='l2', C=1.0, random_state=42)  # L2 정규화
# model.fit(X_train_scaled, y_train)

###################################################################################

# 테스트 데이터로 예측 수행 - 학습된 모델을 사용해 테스트 데이터에 대한 혼잡 여부를 예측 - 참고: Scikit-learn - predict 함수: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.predict
y_pred = model.predict(X_test_scaled)

# 예측 결과 평가 - accuracy_score와 classification_report를 사용해 모델의 정확도 및 성능을 평가
# 참고: Scikit-learn - accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# 참고: Scikit-learn - classification_report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 결과 출력
print(f"모델 정확도: {accuracy:.2f}")
print("분류 리포트:\n", report)

#####

### 앙상블 및 로지스틱 회귀 모델 설정 < 바로 위 소스에서 ####### 안 
# (1) 배깅(bagging) 방식 -  랜덤 포레스트 모델을 생성 / n_estimators=10: 트리의 개수를 10개로 / random_state=42: 결과 재현을 위해 고정된 난수 시드를 설정
# 참고: Scikit-learn - RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)

# (2) 부스팅 방식        - Gradient Boosting 모델을 생성 / n_estimators=10: 트리 개수를 10개로 / learning_rate=0.1: 학습률을 0.1로 설정하여 학습 속도 조절 / max_depth=3: 각 트리의 최대 깊이를 3으로 설정하여 과적합 방지
# 참고: Scikit-learn - GradientBoostingClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
gb_model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=3, random_state=42)

# (3) Logistic Regression - 로지스틱 회귀 모델을 생성 / L2 정규화를 사용 / penalty='l2': L2 정규화(릿지 회귀)를 사용 / C=1.0: 정규화 강도 조절, C 값이 클수록 약한 정규화, 작을수록 강한 정규화 
# 참고: Scikit-learn - LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
lr_model = LogisticRegression(penalty='l2', C=1.0, random_state=42)

### 모델 학습 및 성능 평가 - 설정한 세 가지 모델에 대해 학습을 수행하고, 테스트 데이터에 대한 성능을 평가
models = {'Random Forest': rf_model, 'Gradient Boosting': gb_model, 'Logistic Regression': lr_model}

# 각 모델에 대해 학습 및 평가
for model_name, model in models.items():
    if model_name == 'Logistic Regression':  # Logistic Regression은 스케일링된 데이터를 사용합니다.
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    else:                                    # 앙상블 모델들은 원본 데이터를 사용해 학습합니다.
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # 모델의 정확도와 분류 리포트를 출력합니다.
    # 참고: Scikit-learn - accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    # 참고: Scikit-learn - classification_report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"\n==== {model_name} ====")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{report}")

#####

weekdays_data = pd.read_excel('../data/weekday_traffic.xlsx')

# 독립 변수와 종속 변수 설정 - '8시', '9시', '10시' 열을 독립 변수로 선택하여 교통량 변수를 구성
# 종속 변수: 혼잡 여부 (True/False -> 1/0 변환) - '혼잡' 열을 종속 변수로 사용하며, 이 값이 True이면 1, False이면 0으로 변환
X = weekdays_data[['8시', '9시', '10시']].values
y = weekdays_data['혼잡'].astype(int).values
 
# 데이터 분리  (훈련 데이터 80%, 테스트 데이터 20%) - SVM 모델 학습을 위해 train_test_split을 사용해 데이터를 분리합니다.
# 참고: Scikit-learn - train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

######################
# SVM 모델 생성 및 학습 (훈련 데이터로 학습) - 선형 커널을 사용 / 참고: Scikit-learn - SVC: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
model = SVC()
model.fit(X_train, y_train)

# RandomForest 모델 생성 및 학습 - 배깅 방식 / n_estimators=10: 트리의 개수를 10개로 / random_state=42: 결과 재현을 위해 고정된 난수 시드를 설정
# 참고: Scikit-learn - RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# model = RandomForestClassifier(n_estimators=100)
# model.fit(X_train, y_train)
#######################################


# 예측 수행 - 학습된 모델을 사용해 테스트 데이터에 대한 혼잡 여부를 예측 / 참고: Scikit-learn - predict 함수: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.predict
y_pred = model.predict(X_test)

# 성능 평가 - accuracy_score와 classification_report를 사용해 모델의 정확도 및 성능을 평가 / 참고: Scikit-learn - accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# 참고: Scikit-learn - classification_report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 결과 출력
print(f"모델 정확도: {accuracy:.2f}")
print("분류 리포트:\n", report)

#####