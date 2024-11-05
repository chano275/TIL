from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 추출
driver = webdriver.Chrome()
driver.get("https://seoul-transport-live.kr")

# 대기 시간 설정
wait = WebDriverWait(driver, 10)

# 지하철 데이터 추출
subway_data = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "________")))
subway_df = pd.DataFrame([d.text.split(":") for d in subway_data], columns=["______", "________"])
subway_df["Passengers"] = subway_df["Passengers"].astype(___)

# 버스 데이터 추출
bus_data = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "________")))
bus_df = pd.DataFrame([d.text.split(":") for d in bus_data], columns=["____", "_____"])
bus_df["Count"] = bus_df["Count"].astype(___)

# 시간대별 데이터 추출
time_data = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "________")))
time_df = pd.DataFrame([d.text.split(":") for d in time_data], columns=["____", "_________"])
time_df["Passengers"] = time_df["Passengers"].astype(___)

driver.quit()

# 3-4. 데이터 시각화
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

# 지하철 노선별 이용객 수 막대 그래프
subway_df.plot(x="____", y="__________", kind="___", ax=ax1)
ax1.set_title("Subway Passengers by Line")
ax1.set_xlabel("Subway Line")
ax1.set_ylabel("Number of Passengers")

# 버스 유형별 운행 대수 파이 차트
ax2.pie(bus_df["_____"], labels=bus_df["____"], autopct='%1.1f%%')
ax2.set_title("Bus Operation by Type")

# 시간대별 대중교통 이용객 수 선 그래프
time_df.plot(x="____", y="__________", kind="____", ax=ax3)
ax3.set_title("Public Transport Usage by Time")
ax3.set_xlabel("Time")
ax3.set_ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig("seoul_transport_analysis.png")
plt.close()

# 5. 인사이트 보고서 작성
most_crowded_line = subway_df.loc[subway_df["Passengers"].______(), "Line"]
bus_ratio = bus_df.set_index("Type")["Count"] / bus_df["Count"].___() * 100
peak_time = time_df.loc[time_df["Passengers"].______(), "Time"]

report = f"""
# 서울시 대중교통 이용 현황 분석 보고서

## 주요 인사이트

1. **지하철 이용 현황**
   - 가장 혼잡한 노선: {most_crowded_line}
   - 이 노선은 전체 지하철 이용객의 {subway_df.loc[subway_df["Line"] == most_crowded_line, "Passengers"].values[0] / subway_df["Passengers"].___() * 100:.1f}%를 차지합니다.

2. **버스 운행 현황**
   - 버스 유형별 운행 비율:
     {bus_ratio.to_string()}

3. **시간대별 이용 현황**
   - 대중교통 이용 피크 시간대: {peak_time}
   - 이 시간대의 이용객 수는 {time_df.loc[time_df["Time"] == peak_time, "Passengers"].values[0]:,d}명입니다.

## 결론

이번 분석을 통해 서울시 대중교통의 이용 패턴과 주요 혼잡 구간을 파악할 수 있었습니다. 
{most_crowded_line}의 혼잡도를 줄이기 위한 대책과 {peak_time} 시간대의 대중교통 증편 등을 고려해볼 수 있겠습니다.
또한, 버스 유형별 운행 비율을 참고하여 수요에 맞는 버스 노선 조정이 필요할 것으로 보입니다.
"""

with open("seoul_transport_report.md", "w", encoding="utf-8") as f:
    f._____(report)

print("분석 보고서가 seoul_transport_report.md 파일로 저장되었습니다.")
print("시각화 결과가 seoul_transport_analysis.png 파일로 저장되었습니다.")