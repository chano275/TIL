# 필수 라이브러리 임포트
import pandas as pd
import torch
import nltk
import re
from nltk.tokenize import word_tokenize

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

# ------------------------- 데이터 로드 --------------------------------------------
# CSV 파일에서 데이터를 로드합니다. '제목'과 '교통관련' 열을 사용하여 데이터를 처리할 것입니다.
file_path = '../data/traffic_news.csv'
df = pd.read_csv(file_path)


# ------------------------- 데이터 확인 --------------------------------------------
# 데이터셋에서 '제목'과 '교통관련' 열을 선택하여 모델의 입력과 출력으로 사용합니다.
data = __________________________

# ------------------------- 텍스트 정제 함수 --------------------------------------------
# 텍스트에서 구두점을 제거하고, 모든 문자를 소문자로 변환하여 정제된 텍스트를 만듭니다.
# 참고: https://docs.python.org/3/library/re.html#re.sub
import re  # 정규 표현식 사용을 위한 re 라이브러리

def clean_text(text):
    # re.sub() 함수는 정규 표현식을 사용하여 패턴에 맞는 문자열을 치환합니다. 여기서는 '[^\w\s]' 패턴을 사용하여 알파벳, 숫자, 공백을 제외한 모든 구두점을 제거합니다.
    text = _________________  # 구두점 제거
    text = text.lower()  # 소문자로 변환
    return text

# '제목' 열의 텍스트에 대해 정제 작업을 수행합니다.
data['cleaned_title'] = data['제목'].apply(clean_text)

# ------------------------- 토크나이즈(단어 분리) --------------------------------------------
# NLTK의 word_tokenize 함수를 사용하여 텍스트를 단어 단위로 분리합니다.
# 참고: https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.word_tokenize
from nltk.tokenize import word_tokenize

# 토크나이즈를 하는 이유는 머신러닝 모델에서 텍스트 데이터를 처리할 때, 
# 텍스트를 단어 단위로 나누어야 각 단어가 독립적인 특징(feature)으로 사용될 수 있기 때문입니다.
# 단어 단위로 나눈 후, 각 단어를 벡터화하거나 임베딩을 통해 모델에 입력할 수 있습니다.
# apply 함수와 nltk의 word_tokenize를 통해 를 통해 적용할 수 있습니다.
data['tokenized_title'] = _________________

# 정제된 데이터 확인
print(data[['cleaned_title', 'tokenized_title']].head())

