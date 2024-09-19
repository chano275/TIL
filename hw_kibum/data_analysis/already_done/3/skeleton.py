import pandas as pd

# 1. Titanic 데이터셋에서 'Name', 'Sex', 'Age', 'Pclass', 'Fare' 열만 선택하여 새로운 데이터프레임을 만드세요.
file_path = './titanic.csv'
titanic_data = pd.read_csv(file_path)
selected_columns = titanic_data[['Name', 'Sex', 'Age', 'Pclass', 'Fare']]
# 복사본 생성
titanic_selected = selected_columns.copy()
print(titanic_selected.head())


# # 2. 새로 만든 데이터프레임에 'AgeGroup'이라는 열을 추가하고, 나이를 기준으로 'Child', 'Adult', 'Senior'로 분류하세요.


def classify_age(age):
    if age <= 18:
        return 'Child'
    elif 18 <= age < 65:
        return 'Adult'
    else:
        return 'Senior'


titanic_selected['AgeGroup'] = titanic_selected['Age'].apply(classify_age)  # 슬라이스 복사본에서 작업
print(titanic_selected.head())

# 3. 'Fare' 열에서 요금이 50 이상인 승객들을 추출하여 새로운 데이터프레임을 만드세요.
high_fare_passengers = titanic_selected[titanic_selected['Fare'] >= 50]
print(high_fare_passengers.head())

# 4. 'Pclass' 열의 값이 3인 승객들의 평균 나이(Age)를 계산하세요.
pclass_3_mean_age = titanic_selected[titanic_selected['Pclass'] == 3]['Age'].mean()
print(f"Pclass가 3인 승객들의 평균 나이: {pclass_3_mean_age}")

# 5. 'Sex'가 'female'인 승객들의 평균 'Fare'를 계산하고, 이를 출력하세요.
female_mean_fare = titanic_selected[titanic_selected['Sex'] == 'female']['Fare'].mean()
print(f"Sex가 'female'인 승객들의 평균 Fare: {female_mean_fare}")


##########################



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

import pandas as pd

# 1. Titanic 데이터셋을 로드하세요. 로드한 데이터셋의 첫 5행을 출력하여 데이터의 구조를 확인하세요.
file_path = './titanic.csv'
titanic_data = pd.read_csv(file_path)
print(titanic_data.head(5))

# 2. 'Name', 'Sex', 'Age', 'Pclass', 'Fare' 열만 선택하여 새로운 데이터프레임을 만드세요.
selected_columns = ['Name', 'Sex', 'Age', 'Pclass', 'Fare']
titanic_selected = titanic_data[selected_columns]
print(titanic_selected.head())

# 3. Age 열에서 결측치(NaN)가 있는지 확인하고, 결측치가 있다면 이를 출력하세요.
age_missing = titanic_selected['Age'].isnull().sum()
print(f"Age 열에서 결측치의 수: {age_missing}")

# # 4. Pclass 별로 승객 수를 계산하세요.
pclass_counts = titanic_data.groupby("Pclass").value_counts()
print("Pclass 별 승객 수:")
print(pclass_counts)

# # 5. Fare 열에서 요금이 0보다 큰 승객들만 필터링하여 새로운 데이터프레임을 만드세요.
titanic_filtered = titanic_data[titanic_data['Fare'] > 0]
print("요금이 0보다 큰 승객들:")
print(titanic_filtered.head())
import pandas as pd

# Titanic 데이터셋 로드
file_path = './titanic.csv'
titanic_data = pd.read_csv(file_path)

# 1. Name, Age, Sex 열만 선택하여 새로운 데이터프레임을 만드세요.
selected_columns = titanic_data[['Name', 'Age', 'Sex']]
print(selected_columns.head())

# 2. Age가 30살 이상인 승객만 선택하여 새로운 데이터프레임을 만드세요.
age_30_above = titanic_data[titanic_data['Age'] >= 30]
print(age_30_above.head())

# 3. 3등급(Class 3)에 속한 승객 중 Fare가 20 이하인 승객을 선택하세요.
class_3_fare_20_below = titanic_data[(titanic_data['Pclass'] == 3) & (titanic_data['Fare'] <= 20) ]
print(class_3_fare_20_below.head())

# 4. Age가 40살 이상이고, Pclass가 1인 승객의 Name, Age, Pclass 열을 선택하여 새로운 데이터프레임을 만드세요.
age_40_pclass_1 = titanic_data[(titanic_data['Pclass'] == 1) & (titanic_data['Age'] >= 40)][['Name', 'Age', 'Pclass']]
print(age_40_pclass_1.head())
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
