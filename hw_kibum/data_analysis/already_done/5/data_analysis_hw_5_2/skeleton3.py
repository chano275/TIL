import pandas as pd

# 데이터 불러오기
file_path = ('C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_hw_5_2/processed_data.csv')
data = pd.read_csv(file_path)

# 처리된 데이터 확인
print("결측치 처리 후 데이터셋:")
print(data.isnull().sum())

# 1과 비교해 보면 확실히 줄어든 걸 알 수 있음 