import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 로드
file_path = 'C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_hw_4_2/category_data.csv'
data = pd.read_csv(file_path)

# 2. 서브플롯 생성 및 모든 그래프 그리기
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

# 3. 레이아웃 조정
plt.tight_layout()
plt.show()

