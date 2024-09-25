import requests
import pandas as pd

# API 요청을 위한 정보 설정
api_key = '_________'
date = '_________'  # 원하는 날짜로 변경하세요 (YYYYMMDD 형식)
url = f'http://openapi.seoul.go.kr:8088/{api_key}/json/CardSubwayStatsNew/1/1000/{date}'

# API 요청
response = requests._____(url)
data = response._____

# 데이터가 정상적으로 수신되었는지 확인
if 'CardSubwayStatsNew' in data:
    # 데이터프레임으로 변환
    df = pd.DataFrame(_________)
    # 데이터프레임 출력 (처음 5행)
    print(_________)
else:
    print("데이터를 불러오는데 실패했습니다.")
    print(data)
