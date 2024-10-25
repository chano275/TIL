# 대출 신용정보 평가 (분류, 회귀)

# 대출 신용정보 평가를 위해 분류와 회귀 모델을 모두 사용하는 실습 코드를 준비했습니다. Kaggle에서 유명하고 풍부한 데이터셋인 "Loan Prediction Problem" 데이터셋을 사용하겠습니다.
# 데이터셋 링크: https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset
# 대출 상태를 분류하고 대출 금액을 회귀분석 하는 실습입니다.

# 01. 데이터 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# 데이터 로드
data = pd.read_csv('credit.csv')

# 02. 데이터 전처리
# 필요한 열 선택
data = data[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
             'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
             'Credit_History', 'Property_Area', 'Loan_Status']]

# 결측값 처리
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['Married'].fillna(data['Married'].mode()[0], inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)

# 카테고리형 변수를 숫자로 매핑
le = LabelEncoder()
for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    data[col] = le.fit_transform(data[col])

# 특성과 타겟 분리
X = data.drop(['LoanAmount', 'Loan_Status'], axis=1)
y_class = data['Loan_Status']
y_reg = data['LoanAmount']

# 훈련 데이터와 검증 데이터 분리
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42)

# 03. 데이터 시각화
# 한글 폰트 설정
# !sudo apt-get install -y fonts-nanum
# !sudo fc-cache -fv
# !rm ~/.cache/matplotlib -rf

import matplotlib.pyplot as plt
plt.rc('font', family='NanumBarunGothic')

sampled_data = data.sample(frac=0.1, random_state=42)  # 데이터의 10%를 무작위로 샘플링

def plot_credit_history_vs_loan_status_sampled(data):
    # Credit History와 Loan Status 간의 관계를 빈도 수로 시각화하는 함수
    plt.figure(figsize=(10, 7))
    sns.countplot(x='Credit_History', hue='Loan_Status', data=data, palette='coolwarm')
    plt.title('Credit History와 Loan Status의 관계')
    plt.xlabel('Credit History')
    plt.ylabel('Count')
    plt.show()

def plot_property_area_vs_loan_status_sampled(data):
    # Property Area와 Loan Status 간의 관계를 빈도 수로 시각화하는 함수
    plt.figure(figsize=(10, 7))
    sns.countplot(x='Property_Area', hue='Loan_Status', data=data, palette='coolwarm')
    plt.title('Property Area와 Loan Status의 관계')
    plt.xlabel('Property Area')
    plt.ylabel('Count')
    plt.show()

def plot_income_vs_loan_amount_sampled(data):
    # Applicant Income과 Loan Amount의 관계를 Loan Status에 따라 시각화하는 함수
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='ApplicantIncome', y='LoanAmount', hue='Loan_Status', data=data, palette='coolwarm', alpha=0.6)
    plt.title('Applicant Income과 Loan Amount의 관계 (Loan Status에 따라)')
    plt.xlabel('Applicant Income')
    plt.ylabel('Loan Amount')
    plt.show()

def plot_loan_amount_distribution_sampled(data):
    # Loan Amount의 분포를 히스토그램과 커널 밀도 추정으로 시각화하는 함수
    plt.figure(figsize=(10, 7))
    sns.histplot(data['LoanAmount'].dropna(), kde=True, color='purple')
    plt.title('Loan Amount의 분포')
    plt.xlabel('Loan Amount')
    plt.ylabel('Frequency')
    plt.show()

# 각 시각화 함수 실행
plot_credit_history_vs_loan_status_sampled(sampled_data)
plot_property_area_vs_loan_status_sampled(sampled_data)
plot_income_vs_loan_amount_sampled(sampled_data)
plot_loan_amount_distribution_sampled(sampled_data)

# 04. 분류 머신러닝 모델 학습 및 평가
# LogisticRegression 모델 초기화
lr_model = LogisticRegression()

# lr_model 학습
lr_model.fit(X_train, y_class_train)

# lr_model 예측
y_class_pred = lr_model.predict(X_test)

# RandomForestClassifier 모델 초기화
rf_model = RandomForestClassifier()

# rf_model 학습
rf_model.fit(X_train, y_class_train)

# rf_model 예측
y_class_pred_rf = rf_model.predict(X_test)

# 분류 모델 평가
print("Logistic Regression Classification Report:")
print(classification_report(y_class_test, y_class_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_class_test, y_class_pred))

print("Accuracy Score:")
print(accuracy_score(y_class_test, y_class_pred))

print("\nRandom Forest Classification Report:")
print(classification_report(y_class_test, y_class_pred_rf))

print("Confusion Matrix:")
print(confusion_matrix(y_class_test, y_class_pred_rf))

print("Accuracy Score:")
print(accuracy_score(y_class_test, y_class_pred_rf))

# Feature Importance 계산 및 시각화
feature_importances = rf_model.feature_importances_
feature_names = X.columns

# Feature Importance를 데이터프레임으로 변환
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Feature Importance 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# 05. 회귀 머신러닝 모델 학습 및 평가
# LinearRegression 모델 초기화
lr_reg_model = LinearRegression()

# lr_reg_model 학습
lr_reg_model.fit(X_train, y_reg_train)

# lr_reg_model 예측
y_reg_pred = lr_reg_model.predict(X_test)

# 회귀 모델 평가
print("\nLinear Regression:")

# 평균 제곱 오차 계산 및 출력
mse = mean_squared_error(y_reg_test, y_reg_pred)
print("Mean Squared Error:", mse)

# R-제곱 값 계산 및 출력
r2 = r2_score(y_reg_test, y_reg_pred)
print("R-squared:", r2)

# 회귀 계수 추출 (가중치)
coefficients = lr_reg_model.coef_
feature_names = X_train.columns

# 계수를 데이터프레임으로 변환하여 중요도 정렬
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

# Feature Importance 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='coolwarm')
plt.title('Linear Regression Feature Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()
