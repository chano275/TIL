# 포함 기술스택: python, pandas
import pandas as pd

# 데이터 로드
file_path = 'C:/Users/SSAFY/Desktop/TIL/hw_kibum/data_analysis/data_analysis_hw_6_2/movie_data.csv'
movies = pd.read_csv(file_path)

# 사용자로부터 감독 이름 입력 받기
director_name = input("감독의 이름을 입력하세요: ")

# 특정 감독의 영화 필터링
director_movies = movies[movies['Director'] == director_name]

# 평균 평점 계산
average_rating = director_movies['Rating'].mean()
print(f"{director_name} 감독 영화의 평균 평점: {average_rating:.2f}")

# 가장 높은 평점을 받은 영화 찾기
max_val = movies['Rating'].max()
best_movie = movies[movies['Rating'] == max_val]
for idx, row in best_movie.iterrows():
    print(f"가장 높은 평점을 받은 영화: {row['Title']} ({row['Rating']})")
"""
for 루프를 사용할 때, 변수 b는 **행(row)**이 아니라 **열(column)**을 순환
이로 인해 b['Title']과 같은 방식으로 접근하려고 하면 오류가 발생
best_movie가 데이터프레임일 때, iterrows() 메서드를 사용하여 각 행을 순환하는 방식이 적절
따라서 iterrows()를 사용하여 각 행(row) 데이터를 하나씩 처리
"""