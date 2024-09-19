import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
file_path = ___________
stock_data = ___________

# 7일 이동 평균선 계산
stock_data['7_MA'] = ______________

# 주식 가격이 이동 평균선보다 높은 기간 강조
plt.figure(figsize=(10, 6))
plt.plot(_______, __________, label='Close Price')
plt.plot(_________, _________, label='7-Day MA', linestyle='--')

# 주가가 이동 평균선보다 높은 구간 강조
above_avg = stock_data['Close'] > stock_data['7_MA']
plt.fill_between(__________, ___________, ___________, where=above_avg, color='green', alpha=0.3)

plt.title('Daily Closing Price with Highlighted Periods Above 7-Day MA')
plt.xlabel('Date')
plt.ylabel('Price (KRW)')
plt.xticks(rotation=45)
plt.legend()
plt.show()
