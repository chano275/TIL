import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

file_path = '../data/group_data.csv'
data = pd.read_csv(file_path)

# 특정 조건을 만족하는 데이터 필터링 (예: 특정 열 값이 10에서 20 사이)
filtered_data = data.loc[(data['value'] >= 10) & (data['value'] <= 20)]
print(filtered_data.head())

# 특정 열을 기준으로 그룹화하고 요약 통계 계산 (예: 'category' 열 기준)
grouped_data = filtered_data.groupby('category').describe()
print(grouped_data)

# 특정 열을 기준으로 그룹화하고, 그룹별 평균값 계산
grouped_mean = filtered_data.groupby('category')['value'].mean()
# 평균값을 내림차순으로 정렬하여 출력
sorted_grouped_mean = grouped_mean.sort_values(ascending=False)
print(sorted_grouped_mean)

#################

file_path = '../data/time_series_data.csv'
data = pd.read_csv(file_path, parse_dates=['Date'])
print(data.info())
print(data.describe())

# 특정 기간(예: 2023년 상반기) 필터링
filtered_data = data[(data['Date'] >= '2023-01-01') & (data['Date'] <= '2023-06-30')]
print(filtered_data.head())

# 시간에 따른 변화 시각화
plt.figure(figsize=(12, 6))
plt.plot(filtered_data['Date'], filtered_data['value'], marker='o')
plt.title('Time Series Data (2023 H1)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.show()

# 특정 기간 동안의 평균값과 최대값 계산
mean_value = filtered_data['value'].mean()
max_value = filtered_data['value'].max()
print(f"2023년 상반기 평균값: {mean_value:.2f}")
print(f"2023년 상반기 최대값: {max_value:.2f}")

#################

# 평균이 0이고 표준편차가 1인 정규분포 데이터를 1000개 생성
data = np.random.normal(0, 1, 1000)
print(data[:10])

# 히스토그램 시각화
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor='black')
plt.title('Histogram of Normally Distributed Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# 히스토그램과 KDE 시각화
plt.figure(figsize=(10, 6))
sns.histplot(data, bins=30, kde=True)
plt.title('Histogram with KDE of Normally Distributed Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# 평균과 표준편차 계산
mean = np.mean(data)
std_dev = np.std(data)
print(f"Calculated Mean: {mean:.2f}")
print(f"Calculated Standard Deviation: {std_dev:.2f}")

# 생성된 데이터의 특성과 일치하는지 확인
if np.isclose(mean, 0, atol=0.1) and np.isclose(std_dev, 1, atol=0.1):
    print("The calculated mean and standard deviation are consistent with the generated data's characteristics.")
else:
    print("The calculated mean and standard deviation are not consistent with the generated data's characteristics.")


#################

file_path = '../data/sample_data.csv'
data = pd.read_csv(file_path)
print(data.head())

# 특정 조건을 만족하는 데이터 필터링 (예: 특정 열 값이 10에서 20 사이)
filtered_data = data.loc[(data['column_name'] >= 10) & (data['column_name'] <= 20)]
print(filtered_data.head())

# 특정 열의 히스토그램 시각화
plt.figure(figsize=(10, 6))
plt.hist(filtered_data['column_name'], bins=30, edgecolor='black')
plt.title('Histogram of Filtered Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# 다른 변수와의 관계 시각화 (예: column_name과 다른 변수 간의 관계)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='column_name', y='other_column_name', data=filtered_data)
plt.title('Scatter Plot of Filtered Data')
plt.xlabel('column_name')
plt.ylabel('other_column_name')
plt.show()


#################

file_path = '../data/statistics_data.csv'
data = pd.read_csv(file_path)
print(data.head())

# 기본적인 기술 통계 계산
statistics = data.describe()
print(statistics)

# 기술 통계량 바 차트로 시각화
plt.figure(figsize=(12, 6))
statistics.loc[['mean', '50%', 'min', 'max', 'std']].plot(kind='bar')
plt.title('Basic Statistics of the Dataset')
plt.xlabel('Statistics')
plt.ylabel('Value')
plt.xticks(rotation=0)
plt.show()

# 열 이름 확인
print(data.columns)  # DataFrame의 열 이름을 출력하여 확인

# 히스토그램과 KDE 플롯 시각화
if 'column_name' in data.columns:  # 'column_name' 열이 있는지 확인
    plt.figure(figsize=(12, 6))
    sns.histplot(data['column_name'], kde=True)
    plt.title('Histogram and KDE of column_name')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
else:
    print("Column 'column_name' does not exist in DataFrame. Please check the column names above.")

# 열 이름을 확인한 후, 존재하는 열 이름으로 수정
# 여기서 'column_name'을 실제 존재하는 열 이름으로 바꿔야 함
column_name_to_plot = 'column_name'  # 'column_name'을 데이터셋에 맞는 실제 열 이름으로 변경하세요.

if column_name_to_plot in data.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=data[column_name_to_plot])
    plt.title(f'Boxplot of {column_name_to_plot}')
    plt.ylabel('Value')
    plt.show()
else:
    print(f"Column '{column_name_to_plot}' does not exist in DataFrame. Please check the column names above.")

#################

file_path = '../data/preprocessing_data.csv'
data = pd.read_csv(file_path)
print(data.head())
print(data.info())

# 결측치가 있는 열 확인
missing_data = data.isnull().sum()
print("Missing values in each column:")
print(missing_data[missing_data > 0])

# 결측치 제거 또는 대체 (여기서는 평균값으로 대체 예시)
data['column_with_missing'] = data['column_with_missing'].fillna(data['column_with_missing'].mean())
print(data.info())


# 새로운 파생 변수 생성 (예: 'column_1'의 제곱 값을 새로운 파생 변수로 추가)
data['new_feature'] = data['column_1'].apply(lambda x: x ** 2)
print(data[['column_1', 'new_feature']].head())

# 결측치 처리 및 새로운 파생 변수 생성
data['column_with_missing'] = data['column_with_missing'].fillna(data['column_with_missing'].mean())
data['new_feature'] = data['column_1'].apply(lambda x: x ** 2)  # 'column_1'을 기준으로 새로운 파생 변수 생성

# 데이터 변환 후 요약
summary = data.describe()
print(summary)

# 결측치 처리 전 데이터 저장
before_processing = data.copy()

# 결측치 처리 및 새로운 파생 변수 생성
data['column_with_missing'] = data['column_with_missing'].fillna(data['column_with_missing'].mean())
data['new_feature'] = data['column_1'].apply(lambda x: x ** 2)  # 'column_1'을 기준으로 새로운 파생 변수 생성

# 결측치 처리 후 데이터 비교
print("Before processing:")
print(before_processing.info())
print("\nAfter processing:")
print(data.info())

#################

file_path = '../data/eda_data.csv'
data = pd.read_csv(file_path)
print(data.info())
print(data.describe())

# 'Date' 열을 제외하고 상관관계 계산
correlation_matrix = data.drop(columns=['Date']).corr()

# 상관관계 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 시간에 따른 특정 변수(column_1)의 변화 시각화
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['column_1'], marker='o')
plt.title('Time Series of column_1')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.show()

# 여러 변수 간의 관계를 페어플롯으로 시각화
sns.pairplot(data)
plt.suptitle('Pairplot of Variables', y=1.02)
plt.show()

#################