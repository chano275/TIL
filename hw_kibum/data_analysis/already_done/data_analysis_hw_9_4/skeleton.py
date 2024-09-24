"""
과제 4번 스켈레톤 내에서 구현 안해놓은거 추가해서 올립니당
* 주의 : downloads 폴더 안에 xlsx 파일 ( 실사용하는거 ) 있으면 옮겨놓으시고 사용 부탁드립니당
중간중간에 구현이 귀찮아서 하드코딩한부분때문에 끊길 수 있습니다... 그러면 다시,,, 
"""


import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import folium, openpyxl, os, fnmatch, time


def setup_driver():  # driver 초기세팅 : options 추가 가능
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=options)  # WebDriver가 위에서 설정한 옵션들과 함께 크롬을 실행하도록
    return driver


def navigate_to_page(driver):
    try:
        driver.get("https://topis.seoul.go.kr")
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "contents-area")))
        # 인자로 받은 조건 (EC.presence_of_element_located)이 충족될 때까지, 위에서 설정한 최대 시간(30초) 동안 대기
        # By.ID           : 요소를 찾는 방법을 정의 / HTML 요소의 id 속성에 해당하는 값으로 요소를 찾겠다는 의미
        # "contents-area" : 실제로 찾으려는 HTML 요소의 ID / id="contents-area"인 HTML 요소가 페이지에 존재할 때까지 기다리겠다
        # 페이지 정상적으로 뜬걸 확인 완료
        print("페이지에 접속했습니다.")

    except TimeoutException:
        print("페이지 로드 시간이 초과되었습니다.")
        driver.quit()


def search_keyword(driver, keyword):
    try:
        print(f"'{keyword}' 정류소 정보를 수집합니다.")
        contents_area = driver.find_element(By.ID, "contents-area")                   # 위에서 check 했음
        search_box = contents_area.find_element(By.CSS_SELECTOR, "input.int-search")  # 검색창 선택하고
        search_box.clear()                                                            # 비운 후
        search_box.send_keys(keyword)                                                 # '강남구' 검색
        search_button = contents_area.find_element(By.CSS_SELECTOR, "input.int-btn")  # 엔터도 가능하지만, 버튼 선택
        search_button.click()                                                         # 버튼 클릭
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CLASS_NAME, "asideContent")))  # 위 함수와 동일

        ########################################################################################
        plus_button = driver.find_element(By.CSS_SELECTOR, 'a.btn.btn-default')
        plus_button.click()

        folder_path = 'C:/Users/SSAFY/Downloads'
        file_pattern = "*.xlsx"
        for file_name in os.listdir(folder_path):
            if fnmatch.fnmatch(file_name, file_pattern):
                file_path = os.path.join(folder_path, file_name)
                os.remove(file_path)
                print(f"{file_path} 파일이 삭제되었습니다.")

        download_button = driver.find_element(By.CSS_SELECTOR, "a.bigBtn.btnDgray.line2")
        download_button.click()

        time.sleep(5)  # 개선 가능...

        workbook = openpyxl.load_workbook('C:/Users/SSAFY/Downloads/stationlist.xlsx')
        sheet = workbook.active
        data = []
        for row in sheet.iter_rows(min_row=3, values_only=True):data.append(list(row))
        temp = pd.DataFrame(data[1:], columns=data[0])
        driver.back()

        ########################################################################################

        print(f"'{keyword}' 검색 결과를 찾았습니다.")
        return temp

    except TimeoutException:
        print("검색 결과 로드 시간이 초과되었습니다.")
        driver.quit()


def scrape_bus_stop_data(driver):
    print("정류소 정보를 수집합니다.")
    try:
        aside_content = driver.find_element(By.CLASS_NAME, "asideContent")                                  # 있는거 확인한 정보 가져와서
        results = aside_content.find_element(By.ID, "resultListBusStn").find_elements(By.TAG_NAME, "li")    # 해당 정류소 배열형태로 저장
        # print(results[0].text)
        data = []
        for result in results:
            try:
                name = result.find_element(By.TAG_NAME, "a").text.strip()                                   # 정류소명 (정류장번호) 형태
                # print(name)
                bus_stop_number = name.split("(")[-1].replace(")", "")                                      # (로 나누고, 마지막 ) 를 공백으로 치환 -> 정류소 번호 추출 ( 문자열 형태 )
                # print('*' + bus_stop_number + '*')
                data.append([name, bus_stop_number])
            except NoSuchElementException:
                print("일부 정류소 정보를 찾을 수 없습니다.")

        df = pd.DataFrame(data, columns=['정류소 이름', '정류소 번호'])  # 받아온 데이터들로 df 생성하고 return
        print("정류소 정보가 성공적으로 수집되었습니다.")
        return df
    except NoSuchElementException:
        print("정류소 정보를 찾을 수 없습니다.")
        return pd.DataFrame()
    except TimeoutException:
        print("정류소 정보 수집 중 시간이 초과되었습니다.")
        return pd.DataFrame()
    except Exception as e:
        print(f"정류소 정보 수집 중 오류 발생: {e}")
        return pd.DataFrame()


def visualize_data(df, loc_data):
    if df.empty:
        print("시각화할 데이터가 없습니다.")
        return

    # 임의의 좌표를 사용하여 서울 강남구 중심으로 지도 생성
    map_center = [37.4979, 127.0276]  # 서울 강남구 중심 좌표
    m = folium.Map(location=map_center, zoom_start=13)

    # 정류소를 지도에 마커로 표시
    for index, row in df.iterrows():
        temp_loc = loc_data[loc_data['정류소번호'] == row['정류소 번호']][['Y좌표', 'X좌표']].iloc[0].tolist()

        folium.Marker(
            location=temp_loc,
            popup=row['정류소 이름'],
            tooltip=row['정류소 번호']
        ).add_to(m)

    m.save("bus_stop_map.html")    # 지도 저장
    print("정류소 시각화가 완료되었습니다. 'bus_stop_map.html' 파일을 확인하세요.")


def main():
    driver = setup_driver()
    navigate_to_page(driver)    # 페이지로 이동
    xlsx_data = search_keyword(driver, "강남구")    # 정류소 검색 및 데이터 수집  <<  xlsx_data 에 들어있는 데이터들로 나중에 df에 들어온
    df = scrape_bus_stop_data(driver)
    driver.quit()    # 브라우저 종료

    if not df.empty:    # 데이터 확인 및 시각화
        print("\n수집된 정류소 데이터 분석:")
        print(df.head())
        print(f"총 {len(df)}개의 정류소가 수집되었습니다.")

        # 중복된 정류소 확인
        if df.duplicated(subset=['정류소 번호']).any():
            print("중복된 정류소가 있습니다.")
        else:
            print("중복된 정류소가 없습니다.")

        # 시각화
        visualize_data(df, xlsx_data)
    else:
        print("정류소 데이터 수집에 실패하였습니다.")


if __name__ == "__main__":
    main()
