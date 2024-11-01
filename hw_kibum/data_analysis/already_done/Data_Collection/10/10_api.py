import requests
import pandas as pd
import xml.etree.ElementTree as ET

# 서울시 교통량 API 관련 정보
API_KEY = "6a6e6b734f646f683733635a646e49"  # 주어진 인증키
START_INDEX = 1
END_INDEX = 100  # 최대 100개의 데이터를 한 번에 수집
SERVICE = "SpotInfo"
TYPE = "xml"  # xml 형식으로 데이터 요청

# API 호출 URL
url = f"http://openapi.seoul.go.kr:8088/{API_KEY}/{TYPE}/{SERVICE}/{START_INDEX}/{END_INDEX}/"

# API 호출 및 데이터 수집
response = requests.get(url)

# 응답 상태 코드 확인
if response.status_code == 200:
    try:
        # XML 파싱
        root = ET.fromstring(response.content)

        # 데이터를 저장할 리스트
        data = []

        # XML에서 필요한 데이터를 추출
        for item in root.findall(".//row"):  # XML 구조에 따라 경로 수정
            spot_num = item.find("spot_num").text if item.find("spot_num") is not None else None
            spot_nm = item.find("spot_nm").text if item.find("spot_nm") is not None else None
            grs80tm_x = item.find("grs80tm_x").text if item.find("grs80tm_x") is not None else None
            grs80tm_y = item.find("grs80tm_y").text if item.find("grs80tm_y") is not None else None

            # 데이터를 리스트에 저장
            data.append({
                "SPOT_NUM": spot_num,
                "SPOT_NM": spot_nm,
                "GRS80TM_X": grs80tm_x,
                "GRS80TM_Y": grs80tm_y
            })

        # 데이터프레임으로 변환
        df = pd.DataFrame(data)

        # 데이터 확인(데이터의 처음 5행 출력)
        print("수집한 데이터:")
        print(df.head())

        # 결측치 처리 및 중복 제거
        df_cleaned = df.drop_duplicates().dropna()

        # 피처 엔지니어링: 좌표를 기준으로 '지역구'라는 새로운 피처 생성
        def classify_region(x, y):
            if x < 200000 and y < 450000:
                return "서남권"
            elif x < 200000 and y >= 450000:
                return "서북권"
            elif x >= 200000 and y < 450000:
                return "동남권"
            else:
                return "동북권"

        # 새로운 피처 '지역구' 추가
        df_cleaned['지역구'] = df_cleaned.apply(
            lambda row: classify_region(float(row['GRS80TM_X']), float(row['GRS80TM_Y'])), axis=1)

        # 최종 데이터 출력(데이터의 처음 5행 출력)
        print("정제 및 피처 엔지니어링 적용 후 데이터:")
        print(df_cleaned.head())

    except ET.ParseError as e:
        print(f"XML 파싱 오류: {e}")
else:
    print(f"API 요청 실패. 상태 코드: {response.status_code}")
    print("응답 내용:", response.text)


