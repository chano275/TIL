import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 로드
file_path = 'C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_hw_4_2/category_data.csv'
data = pd.read_csv(file_path)

# 2-1. 가로 막대 그래프 (카테고리별 수치 값)
data_grouped = data.groupby('Category')['Value'].sum()
plt.figure(figsize=(6, 6))# figure size는 6x6으로 설정
plt.barh(data_grouped.index, data_grouped.values, color='skyblue')
plt.title('Category Values')
plt.xlabel('Value')
plt.show()

# 2-2. 파이 차트 (서브 카테고리별 수치 값)
# figure size는 6x6으로 설정
subcategory_data = data.groupby('Subcategory')['Value'].sum()
plt.figure(figsize=(6, 6))
plt.pie(subcategory_data.values, labels=subcategory_data.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
plt.title('Subcategory Distribution')
plt.show()

# # 2-3. 꺾은선 그래프 (카테고리별 수치 값 변화 추이) ?? 
# figure size는 8x6으로 설정
plt.figure(figsize=(8, 6))
for category in data['Category'].unique():
    category_data = data[data['Category'] == category]
    plt.plot(category_data['Subcategory'].sort_values(), category_data['Value'], marker='o', label=category)
plt.title('Category Value Trends')
plt.xlabel('Subcategory')
plt.ylabel('Value')
plt.legend(title='Category')
plt.show()
