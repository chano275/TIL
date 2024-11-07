import pandas as pd
import plotly.express as px

# 1. 데이터 로드 및 병합
f1 = 'C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_ws_4_3/gdp_data.csv'
f2 = 'C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_ws_4_3/population_data.csv'
population_data = pd.read_csv(f2)
gdp_data = pd.read_csv(f1)
merged_data = pd.merge(population_data, gdp_data, on=['Country', 'Year'])

# 2. Plotly를 사용한 점 그래프 생성 및 Hover 기능 추가
fig = px.scatter(merged_data, x='Population', y='GDP'
                , color='Country', size='Population' # 색깔은 나라에 따라 / 사이즈는 인구수에 따라 
                , hover_name='Country'               # 각각의 원에 올렸을때 뜨는 내용 : 나라에 관련된 정보 
                # , log_x=True, size_max=60
)

# 3. 슬라이더 추가
fig.update_layout(title='Population vs GDP Over Time',
                  xaxis_title='Population (Log Scale)',
                  yaxis_title='GDP (USD)',
                  xaxis=dict(tickformat=".0f")  # x축의 숫자 알아보기 쉽게 
                  )

fig.show()
