import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 로드
file_path = 'C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_ws_4_2/survey_data.csv'
data = pd.read_csv(file_path)

# 2. 바이올린 플롯 생성
# sns.catplot(data=data, x='Gender', y='Height', kind='violin')
# sns.violinplot(data=data, x='Gender', y='Height')
# plt.title('Height Distribution by Gender')
# plt.xlabel('Gender')
# plt.ylabel('Height (cm)')
# plt.xticks([0, 1], ['Female', 'Male'])
# plt.show()

# 3. Seaborn 스타일 적용
sns.set(style='whitegrid')
sns.violinplot(data=data, x='Gender', y='Height')
plt.title('Height Distribution by Gender with Seaborn Style')
plt.xlabel('Gender')
plt.ylabel('Height (cm)')
plt.xticks([0, 1], ['Female', 'Male'])
plt.show()
