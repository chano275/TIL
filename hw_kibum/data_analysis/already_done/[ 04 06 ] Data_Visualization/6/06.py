import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc

file_path = '../data/movie_data.csv'
movies = pd.read_csv(file_path)
director_name = input("감독의 이름을 입력하세요: ")
director_movies = movies.loc[movies['Director'] == director_name]

# 평균 평점 계산
average_rating = director_movies['Rating'].mean()
print(f"{director_name} 감독 영화의 평균 평점: {average_rating:.2f}")

# 가장 높은 평점을 받은 영화 찾기
best_movie = director_movies.loc[director_movies['Rating'].idxmax()]
print(f"가장 높은 평점을 받은 영화: {best_movie['Title']} ({best_movie['Rating']})")

#####

file_path = '../data/stock_data.csv'
stock_data = pd.read_csv(file_path)

# 날짜 데이터에서 월 정보 추출
stock_data['Month'] = pd.to_datetime(stock_data['Date']).dt.strftime('%Y-%m')

# 월별 종가 평균 계산
monthly_avg = stock_data.groupby('Month')['Close'].mean()

# 월별 종가 평균 시각화
plt.figure(figsize=(10, 6))
monthly_avg.plot(kind='bar')
plt.title('Monthly Average Closing Price')
plt.xlabel('Month')
plt.ylabel('Average Close Price (KRW)')
plt.xticks(rotation=45)
plt.show()

#####

file_path = '../data/stock_data.csv'
stock_data = pd.read_csv(file_path)

# 날짜 데이터에서 월 정보 추출
stock_data['Month'] = pd.to_datetime(stock_data['Date']).dt.strftime('%Y-%m')

# 월별 종가 평균 계산
monthly_avg = stock_data.groupby('Month')['Close'].mean()

# 월별 종가 평균 시각화
plt.figure(figsize=(10, 6))
monthly_avg.plot(kind='bar')
plt.title('Monthly Average Closing Price')
plt.xlabel('Month')
plt.ylabel('Average Close Price (KRW)')
plt.xticks(rotation=45)
plt.show()

#####

# 한글 폰트 설정
font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"  # MacOS에서의 경로
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

file_path = '../data/movie_data.csv'
movies = pd.read_csv(file_path)
print(movies.head())
spielberg_movies = movies.loc[movies['Director'] == 'Steven Spielberg']
print(spielberg_movies)

# 평균 평점 계산
average_rating = spielberg_movies['Rating'].mean()
print(f"Steven Spielberg 감독 영화의 평균 평점: {average_rating:.2f}")

# 가장 높은 평점을 받은 영화 찾기
best_movie = spielberg_movies.loc[spielberg_movies['Rating'].idxmax()]
print(f"가장 높은 평점을 받은 영화: {best_movie['Title']} ({best_movie['Rating']})")

# 평점 분포 시각화
sns.histplot(spielberg_movies['Rating'], kde=True)
plt.title('Steven Spielberg 감독 영화 평점 분포')
plt.xlabel('평점')
plt.ylabel('영화 수')
plt.show()

#####

file_path = '../data/subway_data.csv'
subway_data = pd.read_csv(file_path)
print(subway_data.head())

# 요일별 승객 수 총합 계산
passengers_by_day = subway_data.groupby('DayOfWeek')['Passengers'].sum()
print(passengers_by_day)

# 요일별 승객 수 변화 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x=passengers_by_day.index, y=passengers_by_day.values)
plt.title('요일별 지하철 승객 수')
plt.xlabel('요일')
plt.ylabel('승객 수 (명)')
plt.show()

# 특정 요일의 평균 승객 수가 가장 높은 역 찾기 (예: 금요일)
day_of_interest = '금'
filtered_data = subway_data[subway_data['DayOfWeek'] == day_of_interest]

# 역별 평균 승객 수 계산
avg_passengers_by_station = filtered_data.groupby('Station')['Passengers'].mean()

# 가장 높은 평균 승객 수를 가진 역 찾기
max_station = avg_passengers_by_station.idxmax()
max_passengers = avg_passengers_by_station.max()
print(f"{day_of_interest}요일에 평균 승객 수가 가장 많은 역: {max_station} ({max_passengers:.0f}명)")

#####

file_path = '../data/stock_data.csv'
stock_data = pd.read_csv(file_path)
print(stock_data.head())

# 일일 종가 데이터 시각화
plt.figure(figsize=(10, 6))
plt.plot(stock_data['Date'], stock_data['Close'], label='Close Price')
plt.title('Daily Closing Price')
plt.xlabel('Date')
plt.ylabel('Close Price (KRW)')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# 7일 이동 평균선 계산
stock_data['7_MA'] = stock_data['Close'].rolling(window=7).mean()

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

# 주식 가격이 이동 평균선보다 높은 기간 강조
plt.figure(figsize=(10, 6))
plt.plot(stock_data['Date'], stock_data['Close'], label='Close Price')
plt.plot(stock_data['Date'], stock_data['7_MA'], label='7-Day MA', linestyle='--')

# 주가가 이동 평균선보다 높은 구간 강조
above_avg = stock_data['Close'] > stock_data['7_MA']
plt.fill_between(stock_data['Date'], stock_data['Close'], stock_data['7_MA'], where=above_avg, color='green', alpha=0.3)
plt.title('Daily Closing Price with Highlighted Periods Above 7-Day MA')
plt.xlabel('Date')
plt.ylabel('Price (KRW)')
plt.xticks(rotation=45)
plt.legend()
plt.show()

#####

file_path = '../data/store_sales.csv'
sales_data = pd.read_csv(file_path)
print(sales_data.head())

# 월별 매출 총합 계산
monthly_revenue = sales_data.groupby('Month')['Revenue'].sum()
print(monthly_revenue)

# 가장 높은 매출을 기록한 달 찾기
max_month = monthly_revenue.idxmax()
max_revenue = monthly_revenue.max()
print(f"가장 높은 매출을 기록한 달: {max_month} (매출: {max_revenue:,} 원)")

# 해당 달의 데이터 필터링
max_month_data = sales_data[sales_data['Month'] == max_month]
# 해당 달에 가장 많이 팔린 제품 찾기
top_product = max_month_data.groupby('Product')['Quantity'].sum().idxmax()
top_quantity = max_month_data.groupby('Product')['Quantity'].sum().max()
print(f"{max_month}에 가장 많이 팔린 제품: {top_product} (판매량: {top_quantity} 개)")


#####

file_path = '../data/movie_reviews.csv'
reviews_data = pd.read_csv(file_path)
print(reviews_data.head())

# 리뷰 길이 계산 및 새로운 열 추가
reviews_data['Review_Length'] = reviews_data['Review'].apply(len)
print(reviews_data[['Review', 'Review_Length']].head())

# 리뷰 길이와 평점 간의 상관관계 계산
correlation = reviews_data['Review_Length'].corr(reviews_data['Rating'])
print(f"리뷰 길이와 평점 간의 상관관계: {correlation:.2f}")

# 리뷰 길이에 따른 평점의 변화를 시각화
plt.figure(figsize=(10, 6))
sns.regplot(x='Review_Length', y='Rating', data=reviews_data, scatter_kws={'alpha':0.3})
plt.title('Review Length vs. Rating')
plt.xlabel('Review Length (characters)')
plt.ylabel('Rating')
plt.show()

# 평균 평점 계산
average_rating = reviews_data['Rating'].mean()

# 특정 길이 이상의 리뷰에서 평점이 평균보다 높은 리뷰 필터링
long_reviews = reviews_data[(reviews_data['Review_Length'] > 200) & (reviews_data['Rating'] > average_rating)]
print(long_reviews[['Review', 'Review_Length', 'Rating']])

#####