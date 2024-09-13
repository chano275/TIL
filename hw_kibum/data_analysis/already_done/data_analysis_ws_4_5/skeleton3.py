import pandas as pd
import plotly.express as px

# 1. 데이터 로드
data = ___________

# 2. 3D 산점도 작성 및 Hover 기능 추가
fig = px.scatter_3d(______________)

# Hover에 추가 정보 표시 (연도)
fig.update_traces(______________)

# 3D 그래프 출력
fig.show()
