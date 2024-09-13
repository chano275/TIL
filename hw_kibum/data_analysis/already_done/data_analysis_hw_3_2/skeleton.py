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
