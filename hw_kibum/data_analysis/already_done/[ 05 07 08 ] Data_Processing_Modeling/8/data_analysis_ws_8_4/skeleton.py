import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# 데이터 로드
df = pd.read_csv('____________')  # 파일 경로를 적절히 변경하세요.

# 데이터 타입 확인
print("문제 1. 데이터의 각 열의 데이터 타입:")
print(df.____________)  # df.dtypes를 사용하여 각 열의 데이터 타입을 확인하세요.

# 날짜 컬럼이 있다면 datetime으로 변환 후 연, 월, 일 등의 정보 추출
if '____________' in df.columns:  # 'date'는 실제 날짜 컬럼 이름으로 변경하세요
    df['date'] = pd.to_datetime(df['____________'])  # 날짜 데이터를 datetime 형식으로 변환합니다.
    df['year'] = df['date'].dt.____________  # 연도 정보 추출
    df['month'] = df['date'].dt.____________  # 월 정보 추출
    df['day'] = df['date'].dt.____________  # 일 정보 추출
    df = df.drop('____________', axis=1)  # 원래 날짜 컬럼은 더 이상 필요 없으므로 삭제
    print("문제 2. 날짜 컬럼을 변환하여 새로운 연, 월, 일 정보가 추가된 데이터 프레임:")
    print(df.____________)  # df.head()로 데이터의 앞부분을 확인하세요.

# 문제 3. 범주형 데이터 처리 - One-Hot Encoding
df = pd.get_dummies(___, __________)  # 데이터프레임과 drop_first=True 옵션을 적절히 입력하세요.

# 독립 변수와 종속 변수 분리
X = df.drop(['____________'], axis=1)  # 'traffic_volume'은 예측하려는 대상 변수입니다.
y = df['____________']

# 데이터 분할
X_train, X_test, y_train, y_test = ___________(X, y, test_size=0.3, random_state=42)  # train_test_split으로 데이터를 분할하세요.
print("\n문제 4. 데이터 분할:")
print(f"훈련 데이터셋 크기: {len(____________)}개, 테스트 데이터셋 크기: {len(____________)}개")  # len() 함수를 사용하여 크기를 확인하세요.

# 문제 5. 모델 훈련 및 평가
model = LinearRegression()
model.fit(____________, ____________)  # 모델 훈련 코드를 입력하세요.
print("\n문제 5. 모델 훈련 완료")

# 예측 및 평가
y_pred = model.predict(____________)  # 예측값을 계산하기 위해 테스트 데이터를 입력하세요.
r_squared = model.score(____________, ____________)  # R-squared를 계산하기 위해 테스트 데이터를 입력하세요.
mse = mean_squared_error(____________, ____________)  # MSE를 계산하기 위해 종속 변수와 예측값을 입력하세요.
mae = mean_absolute_error(____________, ____________)  # MAE를 계산하기 위해 종속 변수와 예측값을 입력하세요.

print("\n문제 6. 모델 평가 지표:")
print(f"R-squared: {____________}")  # R-squared는 모델이 데이터를 얼마나 잘 설명하는지를 나타냅니다.
print(f"MSE: {____________}")  # 평균 제곱 오차(MSE)는 예측값과 실제값 간의 차이를 제곱해 평균을 낸 값입니다.
print(f"MAE: {____________}")  # 평균 절대 오차(MAE)는 예측값과 실제값 간의 절대적 차이의 평균을 나타냅니다.

# 5-fold 교차 검증
cross_val_scores = cross_val_score(model, X, y, cv=5)
average_r2 = np.mean(cross_val_scores)
print(f"\n문제 7. 5-fold 교차 검증 결과:")
print(f"Average R-squared from 5-fold CV: {____________}")  # 평균 R-squared는 여러 세트에서 모델의 일관된 성능을 보여줍니다.

# 결과 해석
print("\n문제 7. 모델의 성능 종합 평가 및 결과 해석:")
print("R-squared 값이 약 94%라는 것은 모델이 _________ 의미합니다.") # 빈칸 채우기
print("MSE와 MAE 값은 모델이 예측하는 값이 실제 값과 얼마나 ________ 있는지를 보여주며, 값이 ______ 예측이 정확합니다.") # 빈칸 채우기
print("5-fold 교차 검증 결과에서 평균 R-squared 값이 높게 나온 것은 모델이 데이터의 패턴을 _______________ 것을 나타냅니다.") # 빈칸 채우기
print("전반적으로, 이 모델은 교통량 예측에 _________, 대부분의 변수들이 모델 성능을 향상시키는 데 _____ 있습니다.") # 빈칸 채우기