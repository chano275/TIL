import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. 데이터 불러오기
# pandas의 read_excel을 사용하여 엑셀 파일에서 데이터를 불러옵니다.
# file_path는 불러올 엑셀 파일의 경로입니다.
# sheet_name은 불러올 시트의 이름을 지정합니다. 여기서는 '2023년 01월' 데이터를 가져옵니다.
# 참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html
file_path = '../data/2023년_01월_서울시_교통량.xlsx'
data = pd.read_excel(file_path, sheet_name='2023년 01월')

# 데이터를 살펴보면 현재 일자와 요일이 일치하지 않습니다. 그렇기 때문에 일자 칼럼을 이용하여 요일을 다시 계산해야 합니다.

# 2. '일자' 컬럼을 datetime 형식으로 변환
# 이를 datetime 형식으로 변환하면 날짜 관련 작업이 더 쉬워집니다.
# pandas의 to_datetime 함수를 사용하여 '일자' 데이터를 날짜 형식으로 변환합니다.
# (참고: '일자' 컬럼은 날짜를 나타내는 값인데, 이 컬럼의 각 값들은 정수 형식(YYYYMMDD)으로 되어 있습니다.)
# 참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html
data['일자'] = pd.to_datetime(data['일자'], format='%Y%m%d')

# 3. 날짜를 기반으로 요일을 자동으로 계산
# 요일 정보는 모델에서 숫자형 데이터로 처리할 수 있도록 변환되어야 합니다.
# datetime 형식으로 변환된 '일자' 컬럼을 기반으로 요일을 계산합니다.
# dt.weekday는 '월요일'=0, '일요일'=6을 반환하므로, +1을 해주어 '월요일'=1, '일요일'=7로 변환합니다.
# 참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.html
data['요일'] = data['일자'].dt.weekday + 1

# 4. 시간대 변수를 그대로 사용
# 기존 데이터셋에서 '0시'부터 '23시'까지 시간대별 교통량 데이터가 이미 존재합니다.
# 이를 독립 변수로 사용하여 각 시간대의 교통량 정보를 피처로 추가합니다.
# 시간대 컬럼은 '0시', '1시'... '23시' 등의 이름으로 존재하므로 그대로 사용할 수 있습니다.

# 5. 데이터 필터링
# '성산로(금화터널)' 지점에서 '유입' 방향으로 들어오는 데이터만 필터링합니다.
# Boolean indexing을 사용하여 조건을 만족하는 행만 선택합니다.
# Boolean indexing: 조건이 True인 데이터만 선택합니다.
# 예: filtered_data = 데이터프레임[(데이터프레임['컬럼명'] == '값') & (데이터프레임['컬럼명'] == '값')]
# 참고: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#boolean-indexing
filtered_data = data[(data['지점명'] == '성산로(금화터널)') & (data['방향'] == '유입')]

# print(filtered_data)

# 6. 피처와 타깃 값 설정
# '요일'과 '시간대' (1시~23시)의 정보를 피처로 사용하고, '0시' 시간대의 교통량을 타깃 값으로 설정합니다.
# 시간대 정보는 이미 데이터셋에서 '1시', '2시' 등의 컬럼으로 존재하므로 이를 독립 변수로 추가합니다.
# pandas의 concat 함수를 사용하여 피처를 결합합니다.
# 참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
# 힌트: 시간대 컬럼은 '1시', '2시', ..., '23시'로 되어 있으며, 이를 for 루프를 통해 추가할 수 있습니다.
# 시간대 컬럼의 이름은 문자열로 처리됩니다. 이를 이용해 여러 열을 동시에 추가할 수 있습니다.
# 예: str(hour) + '시'를 사용하여 1시, 2시 등의 컬럼명을 만들어 보세요.
# '요일'과 '1시'부터 '23시'까지의 데이터를 독립 변수로 사용
X = pd.concat([filtered_data[['요일']], filtered_data[[str(hour) + '시' for hour in range(1, 24)]]], axis=1)
y = filtered_data['0시']  # '0시' 교통량을 종속 변수로 설정



# 7. 학습 데이터와 테스트 데이터 분할
# 학습 데이터와 테스트 데이터를 나누는 함수로 train_test_split을 사용합니다.
# test_size=0.2는 데이터의 20%를 테스트 데이터로 사용한다는 의미입니다.
# random_state=42는 무작위 분할을 고정하여 재현성을 보장합니다.
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. 선형 회귀 모델 정의 및 학습
# scikit-learn의 LinearRegression 모델을 사용하여 학습 데이터를 학습시킵니다.
# 모델을 정의한 후, fit 함수를 사용하여 학습 데이터를 학습합니다.
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
model = LinearRegression()
model.fit(X_train, y_train)

# 9. 모델 평가
# 학습된 모델을 평가하기 위해 R² 점수를 사용합니다. 1에 가까울수록 성능이 좋은 모델입니다.
# score() 함수는 기본적으로 R² 점수를 반환합니다.
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.score
r2_score = model.score(X_test, y_test)
print(f"R²: {r2_score}")
