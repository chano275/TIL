# 필요한 라이브러리 설치
# !pip install torch pandas nltk scikit-learn

# 필수 라이브러리 임포트
import pandas as pd
import torch
import nltk

# ------------------------- NLTK 데이터 다운로드 --------------------------------------------
# NLTK(Natural Language Toolkit) 라이브러리의 punkt 데이터셋을 다운로드합니다.
# punkt는 문장과 단어를 토큰화할 수 있는 도구입니다.
# NLTK 데이터셋을 다운로드하는 함수: https://www.nltk.org/api/nltk.html#nltk.download
nltk.download(_________)

# ------------------------- 장치 설정 --------------------------------------------
# PyTorch에서 사용할 장치를 설정합니다. GPU가 사용 가능하면 CUDA 장치를, 그렇지 않으면 CPU를 사용합니다.
# torch.device() 함수: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
# torch.cuda.is_available() 함수: https://pytorch.org/docs/stable/generated/torch.cuda.is_available.html
device = __________________________
print(f"Using device: {device}")

# ------------------------- 데이터 로드 --------------------------------------------
# CSV 파일에서 데이터를 로드합니다. '제목'과 '교통관련' 열을 사용하여 데이터를 처리할 것입니다.
file_path = '../data/traffic_news.csv'
df = pd.read_csv(file_path)

# 데이터의 열 이름을 확인하여 구조를 파악합니다.
print(df.columns)

# ------------------------- 데이터 확인 --------------------------------------------
# 데이터셋에서 '제목'과 '교통관련' 열을 선택하여 모델의 입력과 출력으로 사용합니다.
data = __________________________

# 데이터의 첫 몇 개의 행을 출력하여 데이터를 확인합니다.
print(data.head())
