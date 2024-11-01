import pandas as pd

file_path = 'data/titanic.csv'
titanic_data = pd.read_csv(file_path)
selected_columns = ['Name', 'Sex', 'Age', 'Pclass', 'Fare']

## 
titanic_selected = titanic_data[selected_columns].copy()

def classify_age(age):
    if age <= 18:    return 'Child'
    elif age <= 60:  return 'Adult'
    else:            return 'Senior'
titanic_selected['AgeGroup'] = titanic_selected['Age'].apply(classify_age)

high_fare_passengers = titanic_selected[titanic_selected['Fare'] >= 50]
pclass_3_mean_age = titanic_selected[titanic_selected['Pclass'] == 3]['Age'].mean()
female_mean_fare = titanic_selected[titanic_selected['Sex'] == 'female']['Fare'].mean()
survived_passengers = titanic_data[titanic_data['Survived'] == 1]
pclass_fare_mean = survived_passengers.groupby('Pclass')['Fare'].mean()
age_30_above = titanic_data[titanic_data['Age'] >= 30]

pclass_1_male_age = titanic_data[(titanic_data['Pclass'] == 1) & (titanic_data['Sex'] == 'male')]['Age']
pclass_1_male_age_mean = pclass_1_male_age.mean()
pclass_1_male_age_median = pclass_1_male_age.median()

embarked_c_count = titanic_data[titanic_data['Embarked'] == 'C'].shape[0]

age_missing = titanic_selected['Age'].isnull().sum()

pclass_counts = titanic_selected['Pclass'].value_counts()

titanic_filtered = titanic_selected[titanic_selected['Fare'] > 0]
class_3_fare_20_below = titanic_data[(titanic_data['Pclass'] == 3) & (titanic_data['Fare'] <= 20)]
age_40_pclass_1 = titanic_data[(titanic_data['Age'] >= 40) & (titanic_data['Pclass'] == 1)][['Name', 'Age', 'Pclass']]
female_under_18 = titanic_data[(titanic_data['Sex'] == 'female') & (titanic_data['Age'] <= 18)]

fare_threshold = titanic_data['Fare'].quantile(0.9)
high_fare_passengers_top10 = titanic_data[titanic_data['Fare'] > fare_threshold]

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

pclass_fare_mean_desc = titanic_data.groupby('Pclass')['Fare'].mean().sort_values(ascending=False)
age_by_survival = titanic_data.groupby('Survived')['Age'].mean()

top_5_fare      = titanic_data.sort_values(by='Fare', ascending=False).head(5)
top_10_youngest = titanic_data.sort_values(by='Age',  ascending=True).head(10)
top_10_pclass_fare = titanic_data.sort_values(by=['Pclass', 'Fare'], ascending=[True, True]).head(10)
top_5_fare_survived = titanic_data[titanic_data['Survived'] == 1].sort_values(by='Fare', ascending=False).head(5)

titanic_data['Weighted_Score'] = titanic_data['Age'] * 0.3 + titanic_data['Fare'] * 0.5 + titanic_data['Pclass'] * 0.2
top_5_weighted = titanic_data.sort_values(by='Weighted_Score', ascending=False).head(5)
