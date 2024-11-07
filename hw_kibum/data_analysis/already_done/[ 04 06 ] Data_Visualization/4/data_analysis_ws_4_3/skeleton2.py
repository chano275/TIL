import pandas as pd
import plotly.express as px

# 1. 데이터 로드 및 병합
f1 = 'C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_ws_4_3/gdp_data.csv'
f2 = 'C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_ws_4_3/population_data.csv'
population_data = pd.read_csv(f2)
gdp_data = pd.read_csv(f1)
merged_data = pd.merge(population_data, gdp_data, on=['Country', 'Year'])

# 2. Plotly를 사용한 점 그래프 생성
fig = px.scatter(merged_data, x='Population', y='GDP')

fig.show()
