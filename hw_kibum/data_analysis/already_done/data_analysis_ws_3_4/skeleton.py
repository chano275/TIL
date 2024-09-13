
import pandas as pd

# Titanic 데이터셋 로드
file_path = './titanic.csv'
titanic_data = pd.read_csv(file_path)

# 1. Age, Fare 열에 대해 평균, 중앙값, 최솟값, 최댓값을 계산하세요.
age_stats = {
    'mean': titanic_data['Age'].mean(),
    'median': titanic_data['Age'].median(),
    'min': titanic_data['Age'].min(),
    'max': titanic_data['Age'].max()
}
fare_stats = {
    'mean': titanic_data['Fare'].mean(),
    'median': titanic_data['Fare'].median(),
    'min': titanic_data['Fare'].min(),
    'max': titanic_data['Fare'].max()
}
print("Age 통계량:", age_stats)
print("Fare 통계량:", fare_stats)

# 2. Pclass별로 평균 Fare를 계산하고, 이를 내림차순으로 정렬하여 출력하세요.
pclass_fare_mean = titanic_data.groupby('Pclass')['Fare'].mean()
print("Pclass별 평균 Fare (내림차순):")
print(pclass_fare_mean)

# 3. 생존 여부에 따른 Age의 평균을 계산하고, 생존자와 비생존자의 나이 평균이 어떻게 다른지 분석하세요.
age_by_survival = titanic_data.groupby('Survived')['Age'].mean()
print("생존 여부에 따른 Age의 평균:")
print(age_by_survival)
