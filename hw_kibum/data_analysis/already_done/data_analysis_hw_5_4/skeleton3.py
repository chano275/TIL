import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 피처 엔지니어링된 데이터 불러오기
data = pd.read_csv('C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_hw_5_4/engineered_data.csv')

# 예측 대상 및 피처 설정
X = data.drop(columns=['CustomerID', 'PurchaseAmount', 'PurchaseDate'])
y = data['PurchaseAmount']


# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # why?


# # 랜덤 포레스트 모델을 사용한 피처 중요도 분석
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# # 피처 중요도 시각화
importances = model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title('Feature Importance')
plt.show()
