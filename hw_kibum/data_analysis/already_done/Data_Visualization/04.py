import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data = pd.read_csv('../data/category_data.csv')

print(data.head())
print(data.describe())

## plt.figure(figsize=(a, b))  # figure size는 a x b로 설정

# 가로 막대 그래프 (카테고리별 수치 값)
data_grouped = data.groupby('Category')['Value'].sum()
plt.figure(figsize=(6, 6))  
plt.barh(data_grouped.index, data_grouped.values, color='skyblue')
plt.title('Category Values')
plt.xlabel('Value')
plt.show()

# 파이 차트 (서브 카테고리별 수치 값)
subcategory_data = data.groupby('Subcategory')['Value'].sum()
plt.figure(figsize=(6, 6))
plt.pie(subcategory_data.values, labels=subcategory_data.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
plt.title('Subcategory Distribution')
plt.show()

# 꺾은선 그래프 (카테고리별 수치 값 변화 추이)
plt.figure(figsize=(8, 6))
for category in data['Category'].unique():
    category_data = data[data['Category'] == category]
    plt.plot(category_data['Subcategory'], category_data['Value'], marker='o', label=category)
plt.title('Category Value Trends')
plt.xlabel('Subcategory')
plt.ylabel('Value')
plt.legend(title='Category')
plt.show()


## 서브플롯 생성 및 모든 그래프 그리기
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 가로 막대 그래프 (카테고리별 수치 값)
data_grouped = data.groupby('Category')['Value'].sum()
axes[0].barh(data_grouped.index, data_grouped.values, color='skyblue')
axes[0].set_title('Category Values')
axes[0].set_xlabel('Value')

# 파이 차트 (서브 카테고리별 수치 값)
subcategory_data = data.groupby('Subcategory')['Value'].sum()
axes[1].pie(subcategory_data.values, labels=subcategory_data.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
axes[1].set_title('Subcategory Distribution')

# 꺾은선 그래프 (카테고리별 수치 값 변화 추이)
for category in data['Category'].unique():
    category_data = data[data['Category'] == category]
    axes[2].plot(category_data['Subcategory'], category_data['Value'], marker='o', label=category)
axes[2].set_title('Category Value Trends')
axes[2].set_xlabel('Subcategory')
axes[2].set_ylabel('Value')
axes[2].legend(title='Category')

###

data = pd.read_csv('../data/statistical_data.csv')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 서브플롯 생성 및 시각화

# 산점도 및 회귀선 시각화
sns.regplot(x='Variable1', y='Variable2', data=data, ax=axes[0])
axes[0].set_title('Scatterplot with Regression Line')
axes[0].set_xlabel('Variable 1')
axes[0].set_ylabel('Variable 2')

# 바이올린 플롯과 커널 밀도 추정(KDE) 플롯을 하나의 서브플롯으로 배치
sns.violinplot(x='Group', y='Variable1', data=data, ax=axes[1])
sns.kdeplot(data['Variable1'], fill=True, ax=axes[1], color='blue')
axes[1].set_title('Violin Plot and KDE of Variable 1')
axes[1].set_xlabel('Group')
axes[1].set_ylabel('Variable 1 / Density')

plt.tight_layout()  # 그래프 요소들이 겹치지 않도록 레이아웃을 자동으로 조정
plt.show()

###

file_path = '../data/seoul_subway_data.csv'
data = pd.read_csv(file_path)

# 요일별 및 시간대별 이용객 수 집계
weekday_agg = data.groupby('DayOfWeek')['PassengerCount'].sum()
time_agg = data.groupby('Time')['PassengerCount'].sum()
print(weekday_agg)
print(time_agg)

# 요일별 이용객 수 집계
weekday_agg = data.groupby('DayOfWeek')['PassengerCount'].sum()
# 요일별 이용객 수 가로 막대 그래프 시각화
plt.figure(figsize=(10, 6))
weekday_agg.sort_values().plot(kind='barh', color='skyblue')
plt.title('Total Passengers by Day of the Week')
plt.xlabel('Passenger Count')
plt.ylabel('Day of the Week')
plt.show()

# 시간대별 이용객 수 집계
time_agg = data.groupby('Time')['PassengerCount'].sum()
# 시간대별 이용객 수 꺾은선 그래프 시각화
plt.figure(figsize=(10, 6))
time_agg.plot(kind='line', color='blue', marker='o')
plt.title('Total Passengers by Time of Day')
plt.xlabel('Time')
plt.ylabel('Passenger Count')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

###

file_path = '../data/survey_data.csv'
data = pd.read_csv(file_path)
print(data.head())
print(data.describe())

# 히스토그램 및 KDE 플롯 생성
plt.figure(figsize=(10, 6))
sns.histplot(data['Height'], kde=True)
plt.title('Height Distribution with Histogram and KDE')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.show()

# 바이올린 플롯 생성
plt.figure(figsize=(10, 6))
sns.violinplot(x='Gender', y='Height', data=data)
plt.title('Height Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Height (cm)')
plt.xticks([0, 1], ['Female', 'Male'])
plt.show()

# Seaborn 스타일 적용
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.violinplot(x='Gender', y='Height', data=data)
plt.title('Height Distribution by Gender with Seaborn Style')
plt.xlabel('Gender')
plt.ylabel('Height (cm)')
plt.xticks([0, 1], ['Female', 'Male'])
plt.show()

###

population_data = pd.read_csv('../data/population_data.csv')
gdp_data = pd.read_csv('../data/gdp_data.csv')

# 데이터프레임 병합
merged_data = pd.merge(population_data, gdp_data, on=['Country', 'Year'])
print(merged_data.head())

# Plotly를 사용한 점 그래프 생성
fig = px.scatter(merged_data, x='Population', y='GDP', color='Country',
                 size='Population', hover_name='Country',
                 log_x=True, size_max=60)
fig.show()

# Plotly를 사용한 점 그래프 생성 및 Hover 기능 추가 + 슬라이더 추가
fig = px.scatter(merged_data, x='Population', y='GDP', color='Country',
                 size='Population', hover_name='Country', animation_frame='Year',
                 log_x=True, size_max=60)
fig.update_layout(title='Population vs GDP Over Time', xaxis_title='Population (Log Scale)', yaxis_title='GDP (USD)', xaxis=dict(tickformat=".0f"))
fig.show()

###

data = pd.read_csv('../data/time_series_data.csv')

# Matplotlib을 사용한 꺾은선 그래프
plt.figure(figsize=(10, 6))
plt.plot(data['Year'], data['Value'], marker='o', linestyle='-')
plt.title('Matplotlib Line Plot')
plt.xlabel('Year')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Seaborn을 사용한 꺾은선 그래프
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Value', data=data, marker='o')
plt.title('Seaborn Line Plot')
plt.xlabel('Year')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Plotly를 사용한 꺾은선 그래프
fig = px.line(data, x='Year', y='Value', title='Plotly Line Plot')
fig.update_traces(mode='markers+lines')

fig.show()

###

data = pd.read_csv('../data/world_data.csv')
print(data.head())

# 3D 산점도 작성
fig = px.scatter_3d(data, x='Population', y='GDP', z='LifeExpectancy',
                    color='Country', size='Population', hover_name='Country')
fig.show()

# 3D 산점도 작성 및 Hover 기능 추가 + Hover에 추가 정보 표시 (연도)
fig = px.scatter_3d(data, x='Population', y='GDP', z='LifeExpectancy',
                    color='Country', size='Population', hover_name='Country')
fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
fig.show()

# 3D 산점도 작성 및 필터 기능 추가
fig = px.scatter_3d(data, x='Population', y='GDP', z='LifeExpectancy',
                    color='Country', size='Population', hover_name='Country',
                    animation_frame='Year', animation_group='Country')
fig.show()

###

