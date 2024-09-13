import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 피처 엔지니어링된 데이터 불러오기
data = pd.read_csv('C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_hw_5_4/engineered_data.csv')

# 예측 대상 및 피처 설정
X = data.drop(columns=['CustomerID', 'PurchaseAmount', 'PurchaseDate'])
y = data['PurchaseAmount']

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # why?

# 랜덤 포레스트 모델 훈련
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 모델 예측 및 성능 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Model Mean Squared Error: {mse}')
