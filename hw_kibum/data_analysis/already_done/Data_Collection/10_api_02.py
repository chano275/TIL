import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# API 요청을 위한 정보 설정
api_key = '51766d5a4e646f6835366971667548'
date = '20230901'  # 원하는 날짜로 변경하세요 (YYYYMMDD 형식)
url = f'http://openapi.seoul.go.kr:8088/{api_key}/json/CardSubwayStatsNew/1/1000/{date}'

# API 요청
response = requests.get(url)
data = response.json()

# 데이터가 정상적으로 수신되었는지 확인
if 'CardSubwayStatsNew' in data:
    df = pd.DataFrame(data['CardSubwayStatsNew']['row'])      # 데이터프레임으로 변환
    print("데이터 수집 완료")
    print(df.head())  
else:
    print("데이터를 불러오는데 실패했습니다.")
    print(data)


df.rename(columns={  # 컬럼명을 한글로 변경 (선택 사항)
    'GTON_TNOPE': '승차총승객수', 'GTOFF_TNOPE': '하차총승객수', 'SBWY_ROUT_LN_NM': '호선명', 'SBWY_STNS_NM': '지하철역'
}, inplace=True)

# 중복 및 이상치 제거
df = df.drop_duplicates()
df = df[(df['승차총승객수'] < 1000000) & (df['하차총승객수'] < 1000000)]
print("중복 및 이상치 제거 완료")

# 데이터에 결측치 추가
df.loc[5, '승차총승객수'] = np.nan   # 5번째 행의 승차총승객수에 결측치 추가
df.loc[10, '하차총승객수'] = np.nan  # 10번째 행의 하차총승객수에 결측치 추가

# 결측치 확인 + 제거 + 처리 후 확인
print("결측치 개수:")
print(df.isnull().sum())
df = df.dropna()
print("결측치 제거 후 결측치 개수:")
print(df.isnull().sum())


# 새로운 피처 생성
df['호선_역명'] = df['호선명'] + '_' + df['지하철역']
print("새로운 피처 '호선_역명' 생성 완료")


# 총승하차인원 계산
df['총승하차인원'] = df['승차총승객수'] + df['하차총승객수']

# 상위 10개 역 추출
top10 = df.sort_values(by='총승하차인원', ascending=False).head(10)

plt.rcParams['font.family'] = 'AppleGothic'  # MacOS에서 한글 폰트 깨짐 방지
# plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows에서 한글 폰트 깨짐 방지

# 상위 10개 역 시각화
plt.bar(top10['지하철역'], top10['총승하차인원'])
plt.xlabel('지하철역')
plt.ylabel('총승하차인원')
plt.title('상위 10개 역의 총승하차인원')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
