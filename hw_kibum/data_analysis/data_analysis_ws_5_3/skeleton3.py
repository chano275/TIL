import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 3. 월별 매출 시계열 분석
file_path = __________
sales_data = __________

# 날짜 데이터를 datetime 형식으로 변환
sales_data['Date'] = ___________

# 매출 (Revenue) 열 계산
sales_data['Revenue'] = _____________

# 월별 매출 계산
sales_data['Month'] = _____________  # 문자열로 변환
monthly_sales = ______________

# 월별 매출 시각화
plt.figure(figsize=(10, 6))
sns.lineplot(______________)
plt.title('Monthly Revenue Trend')
plt.xticks(rotation=45)
plt.show()
