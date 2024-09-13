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
