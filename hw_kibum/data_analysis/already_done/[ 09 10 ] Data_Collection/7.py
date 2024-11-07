import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    return driver


def navigate_to_page(driver):
    driver.get("https://topis.seoul.go.kr")
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID, "contents-area"))
    )


def search_keyword(driver, keyword):
    contents_area = driver.find_element(By.ID, "contents-area")
    search_box = contents_area.find_element(By.CSS_SELECTOR, "input.int-search")
    search_box.send_keys(keyword)
    search_button = contents_area.find_element(By.CSS_SELECTOR, "input.int-btn")
    search_button.click()
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CLASS_NAME, "asideContent"))
    )


def scrape_results(driver):
    aside_content = driver.find_element(By.CLASS_NAME, "asideContent")
    result_sections = {
        "도로": "resultListTraffic",
        "버스": "resultListBus",
        "정류소": "resultListBusStn",
        "따릉이": "resultListBic",
        "주차장": "resultListPark"
    }
    data = {section: [] for section in result_sections.keys()}
    for section_name, result_id in result_sections.items():
        results = aside_content.find_element(By.ID, result_id).find_elements(By.TAG_NAME, "li")
        for result in results:
            item_text = result.text.strip().replace("\n", " | ")  # 줄바꿈을 제거하고 구분자 추가
            data[section_name].append(item_text)
    return data


# 1. 실습4의 데이터를 가져와서 CSV로 저장
def save_data_to_csv(data):
    all_data = [(category, item) for category, items in data.items() for item in items if item != "검색된 내역이 없습니다."]
    df = pd.DataFrame(all_data, columns=["Category", "Item"])
    df.to_csv("seoul_location_data.csv", index=False)
    print("1. 데이터가 seoul_location_data.csv 파일로 저장되었습니다.")


# 2. CSV 파일을 DataFrame으로 불러오기
def load_data_from_csv():
    df = pd.read_csv("seoul_location_data.csv")
    print("\n2. 불러온 DataFrame:")
    print(df.head())
    return df


# 3. 각 카테고리별 데이터 개수 계산
def analyze_data(df):
    counts = df["Category"].value_counts()
    print("\n3. 각 카테고리별 데이터 개수:")
    print(counts)

    # 4. 가장 많은/적은 데이터가 있는 카테고리 찾기
    max_category = counts.idxmax()
    min_category = counts.idxmin()
    print(f"\n4. 가장 많은 데이터가 있는 카테고리: {max_category} ({counts[max_category]}개)")
    print(f"   가장 적은 데이터가 있는 카테고리: {min_category} ({counts[min_category]}개)")

    # 5. 가장 긴/짧은 항목 이름 찾기
    df["Item"] = df["Item"].str.replace("\n", " ")  # CSV에서 줄바꿈을 제거하여 깔끔하게 저장
    longest_name = max(df["Item"], key=len)
    shortest_name = min(df["Item"], key=len)
    print(f"\n5. 가장 긴 이름: {longest_name} ({len(longest_name)}글자)")
    print(f"   가장 짧은 이름: {shortest_name} ({len(shortest_name)}글자)")


def main():
    driver = setup_driver()
    navigate_to_page(driver)
    search_keyword(driver, "관악구")
    data = scrape_results(driver)
    driver.quit()

    save_data_to_csv(data)
    df = load_data_from_csv()
    analyze_data(df)


if __name__ == "__main__":
    main()
