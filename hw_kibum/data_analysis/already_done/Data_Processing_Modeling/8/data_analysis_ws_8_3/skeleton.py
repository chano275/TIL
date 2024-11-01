from sklearn.linear_model import LinearRegression
import pandas as pd

# 데이터 로드
df = pd.read_csv('____________')  # 'seoul_traffic_data.csv' 경로 입력

# 데이터 타입 확인
print("문제 1. 데이터의 각 열의 데이터 타입:")
print(df.____________)  # 각 열의 데이터 타입을 출력

# 날짜 형식이 있는 경우 datetime으로 변환 후, 연, 월, 일 등의 정보 추출
if 'date' in df.columns:  # 'date'는 실제 날짜 컬럼 이름으로 수정하세요
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.____________  # 연도 정보 추출
    df['month'] = df['date'].dt.____________  # 월 정보 추출
    df['day'] = df['date'].dt.____________  # 일 정보 추출
    df = df.drop('____________', axis=1)  # 원래 날짜 컬럼 삭제
    print("문제 1. 날짜 컬럼을 변환하여 새로운 연, 월, 일 정보가 추가된 데이터 프레임:")
    print(df.____________)  # 데이터 프레임의 상위 5개 행 출력

# 범주형 데이터 처리 - One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

# 종속 변수와 독립 변수 분리
X = df.drop(['____________'], axis=1)  # 'traffic_volume'을 타겟 변수로 분리
y = df['____________']  # 종속 변수 설정

# 다중 선형 회귀 모델 구축
model = LinearRegression()
model.fit(X, y)

# 변수의 계수와 R-squared 값 출력
coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': model.coef_})
print("문제 1. 다중 선형 회귀 모델의 각 변수의 계수:")
print(____________)  # 계수 출력
print(f"문제 1. 모델의 R-squared 값: {model.____________(X, y)}")  # R-squared 값 출력

# 영향력 있는 변수 3개 찾기
top_variables = coefficients.nlargest(3, 'Coefficient')
print("문제 2. 가장 영향력 있는 변수 3개:")
print(____________)  # 상위 3개의 변수 출력

# 실습 2의 단순 선형 회귀 모델과 성능 비교
simple_model_r2 = 0.3577104203386625  # 실습 2에서의 단순 회귀 모델 R-squared 값
multi_model_r2 = model.score(X, y)  # 현재 다중 회귀 모델의 R-squared 값

print("\n문제 3. 성능 비교 및 해석:")
print(f"실습 2의 단순 회귀 모델 R-squared 값: {____________}")  # 단순 회귀 R-squared 값 출력
print(f"실습 3의 다중 회귀 모델 R-squared 값: {____________}")  # 다중 회귀 R-squared 값 출력
print("해석: 단순 회귀 모델의 R-squared 값은 약 ______%였지만, 다중 회귀 모델에서는 약 ______%로 성능이 ______되었습니다.") # 빈칸 채우기
print("이는 다중 회귀 모델이 _________ 변수들을 포함하여, 더 많은 데이터의 변화를 설명할 수 있기 때문입니다.") # 빈칸 채우기
