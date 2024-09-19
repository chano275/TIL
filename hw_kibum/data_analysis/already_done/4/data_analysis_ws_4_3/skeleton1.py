import pandas as pd

# 1. 데이터 로드
f1 = 'C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_ws_4_3/gdp_data.csv'
f2 = 'C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_ws_4_3/population_data.csv'
population_data = pd.read_csv(f2)
gdp_data = pd.read_csv(f1)

# print(population_data.info())
# print(gdp_data.info())

# 2. 데이터프레임 병합 - 겹치는 특성 2개 
merged_data = pd.merge(population_data, gdp_data, on=['Country', 'Year'])

# 병합된 데이터프레임 확인
print(merged_data.head())
