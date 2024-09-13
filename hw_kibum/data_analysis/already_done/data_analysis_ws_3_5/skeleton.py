import pandas as pd

# Titanic 데이터셋 로드
file_path = './titanic.csv'
titanic_data = pd.read_csv(file_path)

# 1. Fare를 기준으로 내림차순 정렬한 후, 상위 5명의 승객 정보를 출력하세요.
top_5_fare = titanic_data.sort_values('Fare', ascending=False)
print("Fare 기준 상위 5명:")
print(top_5_fare.head())

# 2. Age가 어린 순으로 정렬한 후, Age 상위 10명을 출력하세요.
top_10_youngest = titanic_data.sort_values('Age', ascending=True)
print("Age 기준 상위 10명 (어린 순):")
print(top_10_youngest.head(10))

# 3. Pclass와 Fare를 기준으로 동시에 오름차순 정렬하고, 상위 10명의 승객 정보를 출력하세요.
top_10_pclass_fare =titanic_data.sort_values(['Pclass', 'Fare'] , ascending=True)
print("Pclass와 Fare 기준 상위 10명:")
print(top_10_pclass_fare.head(10))

# 4. Survived가 1인 승객들 중에서 Fare가 높은 순으로 정렬하여 상위 5명을 출력하세요.
top_5_fare_survived = titanic_data[titanic_data['Survived']==1].sort_values('Fare', ascending=False)
print("생존자 중 Fare 기준 상위 5명:")
print(top_5_fare_survived.head())

# 5. Age, Fare, Pclass를 기준으로 가중 평균을 계산하여 (가중치는 각각 0.3, 0.5, 0.2) 상위 5명을 출력하세요.
titanic_data['Weighted_Score'] = (titanic_data['Age'] * 0.3 + titanic_data['Fare'] * 0.5 + titanic_data['Pclass'] * 0.2)

top_5_weighted = titanic_data.sort_values('Weighted_Score', ascending=False)
print("가중 평균 기준 상위 5명:")
print(top_5_weighted.head())
