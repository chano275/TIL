import pandas as pd

# 데이터 로드
file_path = __________
subway_data = __________

# 특정 요일의 평균 승객 수가 가장 높은 역 찾기 (예: 금요일)
day_of_interest = '금'
filtered_data = __________________

# 역별 평균 승객 수 계산
avg_passengers_by_station = __________________

# 가장 높은 평균 승객 수를 가진 역 찾기
max_station = ____________
max_passengers = ____________

# 결과 출력
print(f"{day_of_interest}요일에 평균 승객 수가 가장 많은 역: {max_station} ({max_passengers:.0f}명)")
