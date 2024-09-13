import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 3. 참여율이 높은 고객과 낮은 고객의 매출 분포 시각화
file_path = _____________
campaign_data = ____________

# 중복된 데이터 제거
campaign_data = _________________

# 참여율이 높은 고객과 낮은 고객의 매출 분포 시각화
high_participation = ________________
low_participation = ______________

plt.figure(figsize=(10, 6))
sns.histplot(____________)
sns.histplot(____________)
plt.title('Revenue Distribution by Participation')
plt.legend()
plt.show()
