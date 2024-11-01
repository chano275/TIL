import requests
import pandas as pd
import xml.etree.ElementTree as ET

# 서울시 교통량 API 관련 정보
API_KEY = "68747344726a6a6335334a674c6154"  # 인증키
START_INDEX, END_INDEX = 1, 100  # 최대 100개의 데이터를 한 번에 수집
SERVICE = "SpotInfo"
TYPE = "xml"  # xml 형식으로 데이터 요청

url = f"http://openapi.seoul.go.kr:8088/{API_KEY}/{TYPE}/{SERVICE}/{START_INDEX}/{END_INDEX}/"  # API 호출 URL
response = requests.get(url)  # API 호출 및 데이터 수집


# 응답 상태 코드 확인
if response.status_code == 200:
    try:
        root = ET.fromstring(response.content)        # XML 파싱
        data = []        # 데이터를 저장할 리스트

        # XML에서 필요한 데이터를 추출
        for item in root.findall(".//row"):  # XML 구조에 따라 경로 수정
            spot_num = item.find("spot_num").text if item.find("spot_num") is not None else None
            spot_nm = item.find("spot_nm").text if item.find("spot_nm") is not None else None
            grs80tm_x = item.find("grs80tm_x").text if item.find("grs80tm_x") is not None else None
            grs80tm_y = item.find("grs80tm_y").text if item.find("grs80tm_y") is not None else None

            # 데이터를 리스트에 저장
            data.append({"SPOT_NUM": spot_num, "SPOT_NM": spot_nm, "GRS80TM_X": grs80tm_x, "GRS80TM_Y": grs80tm_y})

        df = pd.DataFrame(data)        # 데이터프레임으로 변환
        print("수집한 데이터:")
        print(df.head())        # 데이터 확인(데이터의 처음 5행 출력)
        df_cleaned = df.drop_duplicates().dropna()        # 결측치 처리 및 중복 제거

        def classify_region(x, y):        # 피처 엔지니어링: 좌표를 기준으로 '지역구'라는 새로운 피처 생성
            if x < 200000 and y < 450000:                   return "서남권"
            elif x < 200000 and y >= 450000:                return "서북권"
            elif x >= 200000 and y < 450000:                return "동남권"
            else:                                           return "동북권"

        # 새로운 피처 '지역구' 추가
        df_cleaned['지역구'] = df_cleaned.apply(
            lambda row: classify_region(float(row['GRS80TM_X']), float(row['GRS80TM_Y'])), axis=1)

        # 최종 데이터 출력(데이터의 처음 5행 출력)
        print("정제 및 피처 엔지니어링 적용 후 데이터:")
        print(df_cleaned.head())

    except ET.ParseError as e: print(f"XML 파싱 오류: {e}")
else:
    print(f"API 요청 실패. 상태 코드: {response.status_code}")
    print("응답 내용:", response.text)
