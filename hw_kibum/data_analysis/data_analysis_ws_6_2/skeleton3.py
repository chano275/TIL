import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 설정 (MacOS의 경우)
font_path = ____________
font_name = _____________
rc(___________)

# 데이터 로드
file_path = ____________
subway_data = ___________

# 요일별 승객 수 총합 계산
passengers_by_day = ______________

# 요일별 승객 수 변화 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x=passengers_by_day.index, y=passengers_by_day.values)
plt.title('요일별 지하철 승객 수')
plt.xlabel('요일')
plt.ylabel('승객 수 (명)')
plt.show()
