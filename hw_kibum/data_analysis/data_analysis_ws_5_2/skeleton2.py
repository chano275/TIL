import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. 캠페인 성과 비교 및 시각화
file_path = _________
campaign_data = __________

# 중복된 데이터 제거
campaign_data = _______________

# 각 캠페인별 참여율과 평균 클릭률 계산
campaign_summary = ______________

# 캠페인 성과 시각화
sns.barplot(x='CampaignID', y='ParticipationRate', data=campaign_summary)
plt.title('Campaign Participation Rates')
plt.show()
