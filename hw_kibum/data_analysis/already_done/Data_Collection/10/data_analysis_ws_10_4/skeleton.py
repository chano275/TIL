# 실습문제 4: 새로운 피처 생성
import requests
import pandas as pd

# 데이터 수집
api_key = '_________'
date = '20230901' # 데이터 수집 날짜
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

# 새로운 피처 생성
df['호선_역명'] = df['_________'] + '_' + df['_________']
print("새로운 피처 '호선_역명' 생성 완료")
