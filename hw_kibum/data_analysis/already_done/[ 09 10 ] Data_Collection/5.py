import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib3

# SSL 경고 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 1. 웹페이지 가져오기
url = "https://news.seoul.go.kr/traffic/archives/1551"
response = requests.get(url, verify=False)  # SSL 인증서 검증 비활성화
print("문제 1. 웹페이지 가져오기")
print("답: 상태 코드:", response.status_code)

# 2. 모든 테이블 찾기
soup = BeautifulSoup(response.text, 'html.parser')
tables = soup.find_all('table')
print("\n문제 2. 모든 테이블 찾기")
print(f"답: 찾은 테이블 수: {len(tables)}")

# 3. 각 테이블 처리
subway_data = {}
for i, table in enumerate(tables):
    title = table.find_previous('h5')
    table_name = title.text.strip() if title else f"Table {i + 1}"

    data = [[td.text.strip() for td in tr.find_all(['th', 'td'])] for tr in table.find_all('tr')]
    max_cols = max(len(row) for row in data)
    data = [row + [''] * (max_cols - len(row)) for row in data]

    df = pd.DataFrame(data[1:], columns=data[0])
    subway_data[table_name] = df

print("\n문제 3. 테이블 데이터 추출 및 DataFrame 생성")
print(f"답: 처리된 테이블 수: {len(subway_data)}")

# 4. 결과 출력
print("\n문제 4. 각 테이블의 요약 정보 출력")
for name, df in subway_data.items():
    print(f"\n테이블 이름: {name}")
    print("답: 처음 5개 행")
    print(df.head())
    print("\n" + "=" * 50)

# 5. 첫 번째 테이블에서 노선별 정보 추출
print("\n문제 5. 첫 번째 테이블에서 노선별 정보 추출")
if subway_data:
    first_table_name = list(subway_data.keys())[0]
    operation_data = subway_data[first_table_name]

    print("답: 노선별 정보")
    for _, row in operation_data.iterrows():
        if row.iloc[0] and row.iloc[0] != '구 분':
            print(f"\n{row.iloc[0]}:")
            for col, val in row.items():
                if col != row.iloc[0] and val:
                    print(f"  {col}: {val}")

print("\n주의: 이 스크립트는 SSL 인증서 검증을 비활성화했습니다. 실제 환경에서는 보안을 위해 이 방식을 사용하지 않는 것이 좋습니다.")