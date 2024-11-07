from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

def setup_driver():
    # 드라이버 설정 및 브라우저 열기
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")  # 실제 실행할 때는 주석 처리하여 브라우저가 열리도록 설정
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    return driver

def navigate_to_page(driver):
    print("\n문제 1. 페이지 접속 및 로드")
    try:
        # 페이지로 이동
        driver.get("https://topis.seoul.go.kr")
        print("답: 페이지에 접속했습니다.")
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "contents-area"))
        )
        print("답: 페이지가 로드되었습니다.")
    except TimeoutException:
        print("답: 페이지 로딩 시간 초과")
        driver.quit()
    except WebDriverException as e:
        print(f"답: 웹 드라이버 예외 발생: {e}")
        driver.quit()

def search_keyword(driver, keyword):
    print("\n문제 2. 검색 기능 사용")
    try:
        # 검색창 위치를 정확히 찾아 입력
        contents_area = driver.find_element(By.ID, "contents-area")
        search_box = contents_area.find_element(By.CSS_SELECTOR, "input.int-search")
        search_box.send_keys(keyword)
        print(f"답: '{keyword}' 키워드를 입력했습니다.")

        # 검색 버튼 클릭
        search_button = contents_area.find_element(By.CSS_SELECTOR, "input.int-btn")
        search_button.click()
        print("답: 검색 버튼을 클릭했습니다.")

        # 결과가 로딩될 때까지 대기
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CLASS_NAME, "asideContent"))
        )
        print("답: 검색 결과를 찾았습니다.")
    except TimeoutException:
        print("답: 검색 결과 로딩 시간 초과")
    except NoSuchElementException:
        print("답: 검색 요소를 찾을 수 없습니다.")
    except Exception as e:
        print(f"답: 검색 중 오류 발생: {e}")

def scrape_results(driver):
    print("\n문제 3. 검색 결과 크롤링")
    try:
        # 검색 결과 섹션을 찾기
        aside_content = driver.find_element(By.CLASS_NAME, "asideContent")
        result_sections = {
            "도로": "resultListTraffic",
            "버스": "resultListBus",
            "정류소": "resultListBusStn",
            "따릉이": "resultListBic",
            "주차장": "resultListPark"
        }

        # 각 섹션의 검색 결과를 크롤링하여 출력
        for section_name, result_id in result_sections.items():
            try:
                print(f"\n{section_name} 검색 결과:")
                results = aside_content.find_element(By.ID, result_id).find_elements(By.TAG_NAME, "li")
                if results:
                    for result in results:
                        print(f"답: {result.text.strip()}")
                else:
                    print(f"답: {section_name}에서 검색된 내역이 없습니다.")
            except NoSuchElementException:
                print(f"답: {section_name} 검색 결과가 없습니다.")
    except Exception as e:
        print(f"답: 결과 크롤링 중 오류 발생: {e}")

def main():
    driver = setup_driver()
    navigate_to_page(driver)
    search_keyword(driver, "관악구")  # 검색어를 원하는 대로 수정하세요
    scrape_results(driver)
    driver.quit()

if __name__ == "__main__":
    main()
