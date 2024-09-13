
import pandas as pd

# 1. Titanic 데이터셋에서 Survived가 1인 승객들만 필터링하여 새로운 데이터프레임을 만드세요.
file_path = './titanic.csv'
titanic_data = pd.read_csv(file_path)
survived_passengers = titanic_data[titanic_data['Survived'] == 1]
print(survived_passengers.head())

# 2. 위에서 생성한 생존자 데이터프레임에서 Pclass 별로 평균 Fare를 계산하고, 이를 출력하세요.
pclass_fare_mean = survived_passengers.groupby('Pclass')['Fare'].mean()
print("Pclass 별 생존자 평균 Fare:")
print(pclass_fare_mean)

# # 3. Age가 30세 이상인 승객들만 선택하여 새로운 데이터프레임을 만드세요.
age_30_above = titanic_data[titanic_data['Age'] >= 30]
print(age_30_above.head())

# 4. Pclass가 1이고 Sex가 'male'인 승객들의 Age 평균과 중앙값을 계산하세요.
pclass_1_male_age = titanic_data[(titanic_data['Pclass'] == 1) & (titanic_data['Sex'] == 'male')]['Age']
pclass_1_male_age_mean = pclass_1_male_age.mean()
pclass_1_male_age_median = pclass_1_male_age.median()
print(f"Pclass가 1이고 Sex가 'male'인 승객들의 평균 Age: {pclass_1_male_age_mean}")
print(f"Pclass가 1이고 Sex가 'male'인 승객들의 중앙값 Age: {pclass_1_male_age_median}")

# 5. Titanic 데이터셋에서 Embarked 열의 값이 'C'인 승객들의 수를 계산하세요.
embarked_c_count = titanic_data[titanic_data['Embarked'] == 'C'].shape[0]
print(f"Embarked가 'C'인 승객 수: {embarked_c_count}")
