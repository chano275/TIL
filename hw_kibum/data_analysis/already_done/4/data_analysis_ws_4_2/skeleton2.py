import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 로드
file_path = 'C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_ws_4_2/survey_data.csv'
data = pd.read_csv(file_path)

# 2. 히스토그램 및 KDE 플롯 생성
# plt.figure(figsize=(10, 6))

# hist / kde는 displot 안에 
# sns.displot(data=data['Height'], kind='hist')  
# sns.displot(data=data['Height'], kind='kde')

# 다른 그래프지만 이런식으로 가능한듯? hist에 대한 axes 그리고 kde True로
sns.histplot(data['Height'], kde=True)  

plt.title('Height Distribution with Histogram and KDE')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.show()
