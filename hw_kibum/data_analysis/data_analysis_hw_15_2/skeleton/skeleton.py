import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. 데이터 불러오기
# pandas의 read_excel을 사용하여 엑셀 파일에서 데이터를 불러옵니다.
# file_path는 불러올 엑셀 파일의 경로입니다.
# sheet_name은 불러올 시트의 이름을 지정하는데, 여기서는 '2023년 01월' 데이터를 가져옵니다.
# 참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html
file_path = '../data/2023년_01월_서울시_교통량.xlsx'
data = pd.read_excel(file_path, sheet_name='2023년 01월')
# data.dropna(inplace=True)

# 데이터를 살펴보면 현재 일자와 요일이 일치하지 않습니다. 그렇기 때문에 일자 칼럼을 이용하여 요일을 다시 계산해야 합니다.

# 2. '일자' 컬럼을 datetime 형식으로 변환
# 이를 datetime 형식으로 변환하면 날짜 관련 작업이 더 쉬워집니다.
# pandas의 to_datetime 함수를 사용하여 '일자' 데이터를 날짜 형식으로 변환합니다.
# (참고: '일자' 컬럼은 날짜를 나타내는 값인데, 이 컬럼의 각 값들은 정수 형식(YYYYMMDD)으로 되어 있습니다.)
# 참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html
data['일자'] = pd.to_datetime(data['일자'], format='%Y%m%d')

# 3. 날짜를 기반으로 요일을 자동으로 계산
# 요일이 정수로 변환되면 학습 모델에서 이를 숫자형 데이터로 처리할 수 있습니다.
# datetime 형식으로 변환된 '일자' 컬럼을 기반으로 요일을 계산합니다.
# dt.weekday는 '월요일'=0, '일요일'=6을 반환하므로, +1을 해주어 '월요일'=1, '일요일'=7로 변환합니다.
# 참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.html
data['요일'] = data['일자'].dt.weekday + 1
# 4. 데이터 필터링
# '성산로(금화터널)' 지점에서 '유입' 방향으로 들어오는 데이터만을 선택합니다.
# 데이터를 먼저 필터링하여 분석에 필요한 데이터만 남깁니다.
# Boolean indexing을 사용하여 조건에 맞는 데이터만 필터링할 수 있습니다.
# 예: filtered_data = 데이터프레임[데이터프레임['컬럼명'] == '값' & 데이터프레임['컬럼명'] == '값']
# 참고: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#boolean-indexing
filtered_data = data[(data['지점명'] == '성산로(금화터널)') & (data['방향'] == '유입')]

# 5. 피처와 타깃 값 설정
# '요일'을 독립 변수로, '0시' 교통량을 종속 변수로 설정합니다.
# 참고: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
X = filtered_data['요일'].values.reshape(-1, 1)
y = filtered_data['0시']

# 6. 학습 데이터와 테스트 데이터 분할
# 학습 데이터와 테스트 데이터를 나누는 함수로 train_test_split을 사용합니다.
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 선형 회귀 모델 정의 및 학습
# scikit-learn의 LinearRegression 모델을 사용하여 학습 데이터를 학습시킵니다.
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
model = LinearRegression()
model.fit(X_train, y_train)

# 8. 모델 평가
# 학습된 모델을 평가하기 위해 R² 점수를 사용합니다. 1에 가까울수록 좋은 성능을 의미합니다.
# scikit-learn의 score() 함수는 기본적으로 R² 점수를 반환합니다.
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.score
r2_score = model.score(X_test, y_test)
print(f"R²: {r2_score}")
