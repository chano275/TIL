import pandas as pd

# Titanic 데이터셋 로드
file_path = './titanic.csv'
titanic_data = pd.read_csv(file_path)

# 1. 생존한 승객만 필터링하여 Survived 열이 1인 데이터프레임을 만드세요.
survived_passengers = titanic_data[titanic_data['Survived'] == 1]
print(survived_passengers.head())

# 2. Sex가 female이고 Age가 18세 이하인 승객을 필터링하세요.
female_under_18 = titanic_data[(titanic_data['Sex'] == 'female') & (titanic_data['Age'] <= 18)]
print(female_under_18.head())

# 3. Fare가 상위 10%에 속하는 승객을 필터링하세요. 이때 Fare의 상위 10% 기준을 계산하고, 해당 조건을 충족하는 승객 수를 출력하세요.

fare_threshold = titanic_data['Fare'].quantile(0.9)
high_fare_passengers = titanic_data[titanic_data['Fare'] > fare_threshold]
print(f"Fare가 상위 10%에 속하는 승객 수: {high_fare_passengers.shape[0]}")
