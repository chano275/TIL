import pandas as pd

# 데이터 로드
file_path = __________
reviews_data = __________

# 리뷰 길이 계산 및 새로운 열 추가
reviews_data['Review_Length'] = _______________

# 결과 확인
print(reviews_data[['Review', 'Review_Length']].head())
