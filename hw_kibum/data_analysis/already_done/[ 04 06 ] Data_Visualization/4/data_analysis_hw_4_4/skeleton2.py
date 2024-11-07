import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 로드
file_path = 'C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_hw_4_4/statistical_data.csv'
data = pd.read_csv(file_path)

# 2. 서브플롯 생성 및 시각화
fig, axes = plt.subplots(1, 2, figsize=(20, 6))


# 산점도 및 회귀선 시각화
sns.regplot(data=data, x='Variable1', y='Variable2', ax=axes[0])
axes[0].set_title('Scatterplot with Regression Line')
axes[0].set_xlabel('Variable 1')
axes[0].set_ylabel('Variable 2')

# 바이올린 플롯과 커널 밀도 추정(KDE) 플롯을 하나의 서브플롯으로 배치
sns.violinplot(data=data, x='Group', y='Variable1', ax=axes[1])
sns.kdeplot(data['Variable1'], ax=axes[1], fill=True)  
# # kde 플롯 넣으면 바이올린 모양이 안보이는 이유?
axes[1].set_title('Violin Plot and KDE of Variable 1')
axes[1].set_xlabel('Group')
axes[1].set_ylabel('Variable 1 / Density')

# # 3. 레이아웃 조정 및 출력
plt.tight_layout()
plt.show()
