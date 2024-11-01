import pandas as pd

# 데이터 로드
df = pd.read_csv('seoul_traffic_data.csv')  # 파일 경로 입력

print(df.columns)

# 문제 1: 월별 평균 교통량 구하기
df['month'] = pd.to_datetime(df['date']).dt.month  # 날짜를 datetime 형식으로 변환 후, 월을 추출합니다.
monthly_traffic = df.groupby('month')['traffic_volume'].mean().reset_index()  # 월별 평균 교통량을 구합니다.
print("문제 1. 월별 평균 교통량:\n", monthly_traffic)

# 문제 2: 주말 고온 데이터 필터링
high_temp_weekend = df[(df['temperature'] >= 25) & (df['is_weekend'] == 1)]  # 온도가 25도 이상이고 주말인 조건으로 데이터를 필터링합니다.
print("문제 2. 주말 고온 데이터:\n", high_temp_weekend.head())

# 문제 3: 평균 속도가 낮은 데이터 수 구하기
slow_speed_count = len(df[df['avg_speed'] <= 40])  # 평균 속도가 40 이하인 데이터를 선택하고, 그 수를 구합니다.
print("문제 3. 평균 속도가 40 이하인 데이터 수:", slow_speed_count)

# 문제 4: 상관관계 계산
correlation = df['traffic_volume'].corr(df['temperature'])  # 교통량과 기온의 상관관계를 계산합니다.
print("문제 4. 교통량과 기온의 상관관계:", correlation)

# 문제 5: 새로운 변수 생성
df['traffic_density'] = df['traffic_volume'] / df['num_vehicles']  # 새로운 변수 traffic_density를 생성합니다.
print("문제 5. 새로운 변수 traffic_density 생성:\n", df[['traffic_volume', 'num_vehicles', 'traffic_density']].head())
