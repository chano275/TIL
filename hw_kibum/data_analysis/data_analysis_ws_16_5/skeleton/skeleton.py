import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. 데이터 스케일링 (로지스틱 회귀에 반드시 필요)
# 데이터의 스케일에 민감한 로지스틱 회귀 모델에 적용하기 때문에 표준화를 진행합니다. (상대적으로 트리 기반 모델에서는 덜 민감하다는거지 하면 안되는 것은 아닙니다.)
# 앙상블 모델은 스케일링 없이 원본 데이터를 사용합니다.
# 참고: Scikit-learn - StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scaler = StandardScaler()
X_train_scaled = scaler.
X_test_scaled = _____________________________

# 6. 앙상블 및 로지스틱 회귀 모델 설정

# (1) 랜덤 포레스트 - 배깅 방식
# 배깅(bagging) 방식의 랜덤 포레스트 모델을 생성합니다. 
# - n_estimators=10: 트리의 개수를 10개로 설정
# - random_state=42: 결과 재현을 위해 고정된 난수 시드를 설정
# 참고: Scikit-learn - RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
rf_model = RandomForestClassifier(n_estimators=10)  # 트리 몇개

# (2) Gradient Boosting - 부스팅 방식
'''
Gradient Boosting은 여러 개의 약한 학습기(주로 결정 트리)를 순차적으로 학습하여 예측 성능을 향상시키는 앙상블 학습 방법 중 하나입니다. 
각 단계에서 이전 모델이 잘못 예측한 데이터를 더 잘 맞추도록 가중치를 부여하여 새로운 모델을 학습시킵니다. 
최종적으로 모든 약한 학습기의 예측 결과를 합산하여 강력한 예측 모델을 생성합니다.

GradientBoostingClassifier는 이 아이디어를 기반으로 한 분류 모델입니다. 
모델은 먼저 예측이 틀린 데이터에 더 많은 비중을 두고 학습하며, 이를 반복해 나가면서 오차를 줄입니다. 
이 과정은 손실 함수의 기울기를 최소화하는 방식으로 진행되기 때문에 'Gradient'라는 용어가 사용됩니다. 
일반적으로, 과적합을 방지하기 위해 트리의 깊이나 학습률(learning rate) 등을 조절할 수 있습니다.
'''
# 부스팅 방식의 Gradient Boosting 모델을 생성합니다.
# - n_estimators=10: 트리 개수를 10개로 설정
# - learning_rate=0.1: 학습률을 0.1로 설정하여 학습 속도 조절
# - max_depth=3: 각 트리의 최대 깊이를 3으로 설정하여 과적합 방지
# 참고: Scikit-learn - GradientBoostingClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
gb_model = _____________________________

# (3) Logistic Regression - 로지스틱 회귀 모델
# Logistic Regression 모델을 생성하며 L2 정규화를 사용합니다.
# - penalty='l2': L2 정규화(릿지 회귀)를 사용
# - C=1.0: 정규화 강도 조절, C 값이 클수록 약한 정규화, 작을수록 강한 정규화
# 참고: Scikit-learn - LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
lr_model = _____________________________

# 7. 모델 학습 및 성능 평가
# 설정한 세 가지 모델에 대해 학습을 수행하고, 테스트 데이터에 대한 성능을 평가합니다.
models = {
    'Random Forest': _____________________________,
    'Gradient Boosting': _____________________________,
    'Logistic Regression': _____________________________
}

# 각 모델에 대해 학습 및 평가
for model_name, model in models.items():
    if model_name == 'Logistic Regression':
        # Logistic Regression은 스케일링된 데이터를 사용합니다.
        model.fit(X_train_scaled, y_train)
        y_pred = _____________________________
    else:
        # 앙상블 모델들은 원본 데이터를 사용해 학습합니다.
        model.fit(X_train, y_train)
        y_pred = _____________________________
    
    # 모델의 정확도와 분류 리포트를 출력합니다.
    # 참고: Scikit-learn - accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    # 참고: Scikit-learn - classification_report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"\n==== {model_name} ====")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{report}")
