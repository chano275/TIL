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
