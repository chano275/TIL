import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timedelta

# API 키와 기본 URL 설정
BUS_API_KEY = "4d6b675644646f6839356756416248"
SUBWAY_API_KEY = "797352744b646f683131347258726b7a"
BUS_URL = "http://openapi.seoul.go.kr:8088/"
SUBWAY_URL = "http://openapi.seoul.go.kr:8088/"


def get_bus_data(date):    # 버스 데이터를 수집하기 위해 API를 호출
    url = f"{BUS_URL}{BUS_API_KEY}/xml/CardBusStatisticsServiceNew/1/1000/{date}"
    response = requests.get(url)
    print("\n문제 1. 버스 데이터 수집")
    print("답: 상태 코드:", response.status_code)  # 응답 객체의 상태 코드
    print("설명: 상태 코드 200은 API 호출이 성공적으로 완료되었음을 의미합니다.")

    root = ET.fromstring(response.content)    # API 응답 데이터를 XML로 파싱합니다.

    bus_data = []
    for row in root.findall('.//row'):        # XML 데이터에서 필요한 정보 추출해 딕셔너리에 저장
        bus_data.append({
            '날짜': row.find('USE_YMD').text if row.find('USE_YMD') is not None else '',
            '노선번호': row.find('RTE_NM').text if row.find('RTE_NM') is not None else '',
            '정류장명': row.find('STTN_NM').text if row.find('STTN_NM') is not None else '',
            '승차인원': int(row.find('GTON_TNOPE').text) if row.find('GTON_TNOPE') is not None else 0,
            '하차인원': int(row.find('GTOFF_TNOPE').text) if row.find('GTOFF_TNOPE') is not None else 0
        })

    print(f"설명: {date} 날짜의 버스 데이터를 수집하여 리스트에 저장했습니다.")
    return pd.DataFrame(bus_data)


def get_subway_data(date):    # 지하철 데이터를 수집하기 위해 API를 호출합니다.
    url = f"{SUBWAY_URL}{SUBWAY_API_KEY}/xml/CardSubwayStatsNew/1/1000/{date}"
    response = requests.get(url)
    print("\n문제 2. 지하철 데이터 수집")
    print("답: 상태 코드:", response.status_code)  # 응답 객체의 상태 코드
    print("설명: 상태 코드 200은 API 호출이 성공적으로 완료되었음을 의미합니다.")

    root = ET.fromstring(response.content)    # API 응답 데이터를 XML로 파싱합니다.

    subway_data = []
    for row in root.findall('.//row'):        # XML 데이터에서 필요한 정보를 추출해 딕셔너리에 저장
        subway_data.append({
            '날짜': row.find('USE_YMD').text if row.find('USE_YMD') is not None else '',
            '노선번호': row.find('SBWY_ROUT_LN_NM').text if row.find('SBWY_ROUT_LN_NM') is not None else '',
            '역명': row.find('SBWY_STNS_NM').text if row.find('SBWY_STNS_NM') is not None else '',
            '승차인원': int(row.find('GTON_TNOPE').text) if row.find('GTON_TNOPE') is not None else 0,
            '하차인원': int(row.find('GTOFF_TNOPE').text) if row.find('GTOFF_TNOPE') is not None else 0
        })

    print(f"설명: {date} 날짜의 지하철 데이터를 수집하여 리스트에 저장했습니다.")
    return pd.DataFrame(subway_data)


def integrate_data(bus_data, subway_data):    # 버스와 지하철 데이터를 통합합니다.
    bus_data['교통수단'] = '버스'
    bus_data = bus_data.rename(columns={'정류장명': '역/정류장명'})

    subway_data['교통수단'] = '지하철'
    subway_data = subway_data.rename(columns={'역명': '역/정류장명'})

    # pd.concat()을 사용하여 두 개의 DataFrame을 하나로 합칩니다.
    integrated_df = pd.concat([bus_data, subway_data], ignore_index=True)
    print("\n문제 3. 데이터 통합")
    print("답: 버스와 지하철 데이터가 통합되었습니다.")
    print(f"설명: 통합된 데이터의 컬럼은 {integrated_df.columns.tolist()} 입니다.")
    return integrated_df[['교통수단', '날짜', '노선번호', '역/정류장명', '승차인원', '하차인원']]


def get_data_for_date_range(start_date, end_date):    # 지정된 날짜 범위에 대해 데이터를 수집하고 통합합니다.
    current_date = start_date
    all_data = []

    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        print(f"\n설명: {date_str} 날짜의 데이터를 수집 중입니다.")

        bus_data = get_bus_data(date_str)
        subway_data = get_subway_data(date_str)

        daily_data = integrate_data(bus_data, subway_data)
        all_data.append(daily_data)

        current_date += timedelta(days=1)

    return pd.concat(all_data, ignore_index=True)


def main():
    # 데이터를 수집할 시작 날짜와 종료 날짜를 설정합니다.
    start_date = datetime(2023, 9, 1)
    end_date = datetime(2023, 9, 7)  # 일주일 치 데이터

    all_data = get_data_for_date_range(start_date, end_date)    # 지정된 날짜 범위에 대한 데이터를 수집하고 통합합니다.

    daily_totals = all_data.groupby(['교통수단', '날짜']).agg({    # 일별 총 승하차 인원을 계산
        '승차인원': 'sum',
        '하차인원': 'sum'
    }).reset_index()
    print("\n문제 4. 일별 총 승하차 인원 계산")
    print(daily_totals.head())
    print("설명: 일별로 교통수단별 총 승하차 인원을 계산하여 데이터프레임에 저장했습니다.")

    route_averages = all_data.groupby(['교통수단', '노선번호']).agg({    # 노선별 평균 승하차 인원 계산
        '승차인원': 'mean',
        '하차인원': 'mean'
    }).reset_index()
    print("\n문제 5. 노선별 평균 승하차 인원 계산")
    print(route_averages.head())
    print("설명: 노선별로 평균 승하차 인원을 계산하여 데이터프레임에 저장했습니다.")

    # 결과를 CSV 파일로 저장
    all_data.to_csv("seoul_transport_data_detailed.csv", index=False, encoding='utf-8-sig')
    daily_totals.to_csv("seoul_transport_daily_totals.csv", index=False, encoding='utf-8-sig')
    route_averages.to_csv("seoul_transport_route_averages.csv", index=False, encoding='utf-8-sig')
    print("\n문제 6. 결과 CSV 파일로 저장")
    print("설명: 데이터를 분석하고, CSV 파일로 저장하여 나중에 참고할 수 있도록 했습니다.")
    print("Data has been saved to CSV files.")


if __name__ == "__main__":
    main()
