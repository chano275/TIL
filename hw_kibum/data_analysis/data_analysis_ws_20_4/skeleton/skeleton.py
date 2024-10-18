from collections import Counter
import torch
import re
from nltk.tokenize import word_tokenize
import pandas as pd
import torch
import nltk
from sklearn.model_selection import train_test_split


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


# ------------------------- 단어에 고유한 정수 인덱스 부여 --------------------------------------------
# 단어 토큰 리스트에서 단어에 고유한 정수 인덱스를 부여하는 작업을 수행합니다.
# 각 단어를 고유한 정수로 변환하여 텍스트 데이터를 모델이 처리할 수 있도록 준비합니다.
# 이를 위해 단어의 빈도를 계산한 후, 빈도 순으로 단어에 인덱스를 부여합니다.

# 단어 빈도를 계산합니다. 
# tokenized_texts 리스트에서 각 단어가 얼마나 자주 등장하는지 카운트합니다.
def build_vocab(tokenized_texts):
    # Counter는 각 단어의 빈도를 세는 데 사용되며, 이를 통해 단어의 출현 빈도를 알 수 있습니다.
    # 텍스트 리스트 안의 각 단어를 풀어 헤치고 빈도수를 계산
    word_count = ______________________

    # 빈도순으로 정렬된 단어에 대해 고유한 인덱스를 부여합니다.
    # 빈도가 높은 단어부터 인덱스를 부여하고, 인덱스는 1부터 시작합니다.
    vocab = ______________________

    # 생성된 단어 사전(vocab)을 반환합니다. 각 단어는 고유한 인덱스를 가집니다.
    return vocab

# 'tokenized_title'에서 단어 사전을 구축합니다.
vocab = build_vocab(data['tokenized_title'])

# ------------------------- 텍스트를 정수 인덱스 시퀀스로 변환 --------------------------------------------
# 각 단어를 고유한 정수 인덱스로 변환하여 시퀀스를 만듭니다. 모델에 입력으로 사용할 수 있게 고정된 길이로 변환합니다.
# 시퀀스가 최대 길이를 초과하면 잘라내고, 짧으면 0으로 패딩(빈 공간 채우기)하여 고정 길이로 맞춥니다.
def text_to_sequence(tokenized_text, vocab, max_len=20):
    # 단어 리스트를 사전에 있는 단어에 대응하는 정수 인덱스로 변환합니다.
    # vocab에 없는 단어는 0으로 처리합니다.
    # vocab.get(word, 0): 단어가 vocab에 있으면 그 단어의 인덱스를, 없으면 0을 반환합니다.
    # TODO: 리스트 컴프리헨션이나 for문을 통해 sequence를 만들어주세요.
    sequence = _______________________
    
    # 시퀀스가 최대 길이보다 짧으면 0으로 패딩하여 길이를 맞춥니다.
    # 최대 길이보다 짧은 경우 (max_len - len(sequence))만큼 0을 추가해 고정 길이를 유지합니다.
    if len(sequence) < max_len:
        sequence += [0] * (max_len - len(sequence))
    
    # 정수 인덱스 시퀀스를 반환합니다.
    return sequence

# 'tokenized_title' 열의 각 행에 대해 text_to_sequence 함수를 직접 적용합니다.
sequences = []
for index, row in data['tokenized_title'].items():
    sequences.append(_______________________)

# 변환된 시퀀스를 새로운 열로 추가합니다.
data['text_sequence'] = sequences

# 데이터 확인
print(data[['tokenized_title', 'text_sequence']].head())

# ------------------------- 텐서 변환 및 데이터 분할 --------------------------------------------
# 데이터를 PyTorch에서 사용할 수 있도록 텐서로 변환합니다. 텍스트 시퀀스(X)와 라벨(y)을 텐서로 변환하여 학습에 사용할 수 있게 만듭니다.
# torch.tensor: 리스트나 배열을 PyTorch 텐서로 변환합니다. 모델 학습에서 사용할 수 있도록 시퀀스 데이터를 텐서로 만듭니다.
X = torch.tensor(data['text_sequence'].tolist())  # 각 시퀀스를 텐서로 변환
y = torch.tensor(data['교통관련'].values)  # 라벨 데이터를 텐서로 변환

# 데이터를 훈련셋과 테스트셋으로 분할합니다. 이 과정은 모델이 새로운 데이터를 평가할 수 있도록 도와줍니다.
# train_test_split 함수는 데이터를 80:20 비율로 나누어 훈련셋과 테스트셋을 생성합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 훈련 데이터의 크기를 확인하여 데이터가 정상적으로 처리되었는지 확인합니다.
# 데이터가 훈련셋으로 나누어졌고, 각각의 텐서 크기가 예상대로인지 확인합니다.
print(f"훈련 데이터 크기: {X_train.shape}")
