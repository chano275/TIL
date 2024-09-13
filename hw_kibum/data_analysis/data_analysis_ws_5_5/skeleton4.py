import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 4. 피처 중요도 분석
data = _____________

# 예측 대상 및 피처 설정
X = data.drop(columns=['ProductID', 'SalesVolume'])
y = data['SalesVolume']

# 데이터셋 분할
X_train, X_test, y_train, y_test = ____________

# 랜덤 포레스트 모델을 사용한 피처 중요도 분석
model = _______________
model.fit(X_train, y_train)

# 피처 중요도 시각화
importances = __________
features = _________
sns.barplot(x=importances, y=features)
plt.title('Feature Importance')
plt.show()

# 모델 성능 평가
y_pred = model.predict(X_test)
mse = __________(y_test, y_pred)
print(f'Model Mean Squared Error: {mse}')
