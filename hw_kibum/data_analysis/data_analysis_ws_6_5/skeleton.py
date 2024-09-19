import pandas as pd

# 데이터 로드
file_path = ____________
reviews_data = ___________

# 리뷰 길이 계산 및 새로운 열 추가
reviews_data['Review_Length'] = ____________

# 평균 평점 계산
average_rating = ____________

# 특정 길이 이상의 리뷰에서 평점이 평균보다 높은 리뷰 필터링
long_reviews = ________________

# 필터링된 리뷰 출력
print(long_reviews[['Review', 'Review_Length', 'Rating']])
