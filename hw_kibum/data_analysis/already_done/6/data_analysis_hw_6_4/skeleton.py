import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
file_path = 'C:/Users/SSAFY/Desktop/TIL/hw_kibum/data_analysis/data_analysis_hw_6_4/stock_data.csv'
stock_data = pd.read_csv(file_path)

# print(stock_data.head())

stock_data['Month'] = pd.to_datetime(stock_data['Date']).dt.strftime('%m')# 날짜 데이터에서 월 정보 추출
monthly_avg = stock_data.groupby('Month')['Close'].mean()# 월별 종가 평균 계산

# print(monthly_avg)

# # 월별 종가 평균 시각화
plt.figure(figsize=(10, 6))
monthly_avg.plot()
plt.title('Monthly Average Closing Price')
plt.xlabel('Month')
plt.ylabel('Average Close Price (KRW)')
plt.xticks(rotation=45)
plt.show()
