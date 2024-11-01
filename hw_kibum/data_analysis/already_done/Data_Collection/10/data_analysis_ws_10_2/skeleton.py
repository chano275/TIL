import requests
import pandas as pd

# 실습문제 1: 데이터 수집
api_key = '_________'
date = '_________'
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

# 실습문제 2: 중복 및 이상치 제거
df = df._________()  # 중복 제거
df = df[(df['승차총승객수'] < _________) & (df['하차총승객수'] < _________)]  # 이상치 제거
print("중복 및 이상치 제거 완료")
