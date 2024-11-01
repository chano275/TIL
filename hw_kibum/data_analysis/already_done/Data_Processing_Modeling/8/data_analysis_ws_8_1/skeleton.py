import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드 및 기본 정보 출력
df = pd.read_csv('____________')  # 파일 경로 입력
print("문제 1. 데이터의 구조와 각 열의 데이터 타입:")
print(df.____________())  # 데이터 타입 확인
print("문제 1. 데이터의 첫 5행 출력:")
print(df.____________())  # 데이터 첫 5행 출력

# 산점도 시각화
plt.figure(figsize=(10, 6))
plt.scatter(df['____________'], df['____________'], alpha=0.5)  # X, y 설정
plt.title('____________')
plt.xlabel('____________')
plt.ylabel('____________')
plt.grid(True)
plt.show()

# 관계 해석
print("문제 3. 해석: 교통량이 증가할수록 평균 속도가 _____하는 경향이 보입니다.")  # 빈칸 채우기
