import pandas as pd

# 데이터 로드
file_path = __________
reviews_data = __________

# 리뷰 길이 계산 및 새로운 열 추가
reviews_data['Review_Length'] = ____________

# 리뷰 길이와 평점 간의 상관관계 계산
correlation = ______________

# 상관관계 출력
print(f"리뷰 길이와 평점 간의 상관관계: {correlation:.2f}")
