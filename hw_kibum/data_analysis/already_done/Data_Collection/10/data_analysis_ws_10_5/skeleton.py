# 실습문제 5: 상위 10개 역 시각화
import requests
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 수집
api_key = '_________'
date = '20230901'
url = f'http://openapi.seoul.go.kr:8088/{api_key}/json/CardSubwayStatsNew/1/1000/{date}'

response = requests.get(url)
data = response._____

if 'CardSubwayStatsNew' in data:
    df = pd.DataFrame(data['CardSubwayStatsNew']['row'])
    print("데이터 수집 완료")
else:
    print("데이터를 불러오는데 실패했습니다.")
    print(data)
    exit()

# 컬럼명을 한글로 변경 (선택 사항)
df.rename(columns={
    'GTON_TNOPE': '승차총승객수',
    'GTOFF_TNOPE': '하차총승객수',
    'SBWY_ROUT_LN_NM': '호선명',
    'SBWY_STNS_NM': '지하철역'
}, inplace=True)

# 총승하차인원 계산
df['총승하차인원'] = df['_________'] + df['_________']

# 상위 10개 역 추출
top10 = df.sort_values(by='_________', ascending=False).head(_________)

plt.rcParams['font.family'] = 'AppleGothic'  # MacOS에서 한글 폰트 깨짐 방지
# plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows에서 한글 폰트 깨짐 방지

# 상위 10개 역 시각화
plt.bar(top10['_________'], top10['_________'])
plt.xlabel('지하철역')
plt.ylabel('총승하차인원')
plt.title('상위 10개 역의 총승하차인원')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
