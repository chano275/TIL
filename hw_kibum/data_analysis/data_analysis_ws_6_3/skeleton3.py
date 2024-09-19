import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
file_path = ____________
stock_data = ____________

# 7일 이동 평균선 계산
stock_data['7_MA'] = stock_data['Close'].rolling(window=_________).mean()

# 일일 종가 및 7일 이동 평균선 시각화
plt.figure(figsize=(10, 6))
plt.plot(stock_data['Date'], stock_data['Close'], label='Close Price')
plt.plot(stock_data['Date'], stock_data['7_MA'], label='7-Day MA', linestyle='--')
plt.title('Daily Closing Price with 7-Day Moving Average')
plt.xlabel('Date')
plt.ylabel('Price (KRW)')
plt.xticks(rotation=45)
plt.legend()
plt.show()
