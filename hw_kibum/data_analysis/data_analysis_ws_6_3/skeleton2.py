import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
file_path = ___________
stock_data = ___________

# 일일 종가 데이터 시각화
plt.figure(figsize=(10, 6))
plt.plot(___________________)
plt.title('Daily Closing Price')
plt.xlabel('Date')
plt.ylabel('Close Price (KRW)')
plt.xticks(rotation=45)
plt.legend()
plt.show()
