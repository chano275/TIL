# 실습문제 3: 결측치 처리
import requests
import pandas as pd
import numpy as np

# 데이터 수집
api_key = '_________'
date = '20230901'
url = f'http://openapi.seoul.go.kr:8088/{api_key}/json/CardSubwayStatsNew/1/1000/{date}'

response = requests.get(url)
data = response.json()

if 'CardSubwayStatsNew' in data:
    df = pd.DataFrame(data['CardSubwayStatsNew']['row'])
    print("데이터 수집 완료")
else:
    print("데이터를 불러오는데 실패했습니다.")
    print(data)
    exit()

# 컬럼명 한글로 변경
df.rename(columns={
    'GTON_TNOPE': '승차총승객수',
    'GTOFF_TNOPE': '하차총승객수',
    'SBWY_ROUT_LN_NM': '호선명',
    'SBWY_STNS_NM': '지하철역'
}, inplace=True)

# 데이터에 결측치 추가
# 5번째 행의 승차총승객수에 결측치 추가
df.loc[5, '승차총승객수'] = np.nan
# 10번째 행의 하차총승객수에 결측치 추가
df.loc[10, '하차총승객수'] = np.nan

# 결측치 확인
print("결측치 개수:")
print(df.____().____())

# 결측치 제거
df = df.____()

# 결측치 제거 후 다시 확인
print("결측치 제거 후 결측치 개수:")
print(df.____().____())
