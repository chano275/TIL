# 필요한 라이브러리 설치 pip install torch pandas nltk scikit-learn

import pandas as pd
import numpy as np
import torch, nltk, re, random
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.model_selection import train_test_split
from torch import nn, optim


# ------------------------- NLTK 데이터 다운로드 --------------------------------------------
# NLTK(Natural Language Toolkit) 라이브러리의 punkt 데이터셋을 다운로드합니다.
# punkt는 문장과 단어를 토큰화할 수 있는 도구입니다.
# NLTK 데이터셋을 다운로드하는 함수: https://www.nltk.org/api/nltk.html#nltk.download
nltk.download('punkt')

# ------------------------- 장치 설정 --------------------------------------------
# PyTorch에서 사용할 장치를 설정합니다. GPU가 사용 가능하면 CUDA 장치를, 그렇지 않으면 CPU를 사용합니다.
# torch.device() 함수: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
# torch.cuda.is_available() 함수: https://pytorch.org/docs/stable/generated/torch.cuda.is_available.html
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------- 데이터 로드 --------------------------------------------
# CSV 파일에서 데이터를 로드합니다. '제목'과 '교통관련' 열을 사용하여 데이터를 처리할 것입니다.
file_path = '../data/traffic_news.csv'
df = pd.read_csv(file_path)

# 데이터의 열 이름을 확인하여 구조를 파악합니다.
print(df.columns)

# ------------------------- 데이터 확인 --------------------------------------------
# 데이터셋에서 '제목'과 '교통관련' 열을 선택하여 모델의 입력과 출력으로 사용합니다.
data = df[['제목', '교통관련']]

# 데이터의 첫 몇 개의 행을 출력하여 데이터를 확인합니다.
print(data.head())


# ------------------------- 텍스트 정제 함수 --------------------------------------------
# 텍스트에서 구두점을 제거하고, 모든 문자를 소문자로 변환하여 정제된 텍스트를 만듭니다.
# 참고: https://docs.python.org/3/library/re.html#re.sub
import re  # 정규 표현식 사용을 위한 re 라이브러리

def clean_text(text):
    # re.sub() 함수는 정규 표현식을 사용하여 패턴에 맞는 문자열을 치환합니다. 여기서는 '[^\w\s]' 패턴을 사용하여 알파벳, 숫자, 공백을 제외한 모든 구두점을 제거합니다.
    text = re.sub(r'[^\w\s]', '', text)  # 구두점 제거
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
# apply 함수를 통해 적용할 수 있습니다.
data['tokenized_title'] = data['cleaned_title'].apply(word_tokenize)

# 정제된 데이터 확인
print(data[['cleaned_title', 'tokenized_title']].head())


# ------------------------- 단어에 고유한 정수 인덱스 부여 --------------------------------------------
# 단어 토큰 리스트에서 단어에 고유한 정수 인덱스를 부여하는 작업을 수행합니다.
# 각 단어를 고유한 정수로 변환하여 텍스트 데이터를 모델이 처리할 수 있도록 준비합니다.
# 이를 위해 단어의 빈도를 계산한 후, 빈도 순으로 단어에 인덱스를 부여합니다.

# 단어 빈도를 계산합니다. 
# tokenized_texts 리스트에서 각 단어가 얼마나 자주 등장하는지 카운트합니다.
def build_vocab(tokenized_texts):
    # Counter는 각 단어의 빈도를 세는 데 사용되며, 이를 통해 단어의 출현 빈도를 알 수 있습니다.
    # 텍스트 리스트 안의 각 단어를 풀어 헤치고 빈도수를 계산
    word_count = Counter([word for text in tokenized_texts for word in text])

    # 빈도순으로 정렬된 단어에 대해 고유한 인덱스를 부여합니다.
    # 빈도가 높은 단어부터 인덱스를 부여하고, 인덱스는 1부터 시작합니다.
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_count.most_common())}

    # 생성된 단어 사전(vocab)을 반환합니다. 각 단어는 고유한 인덱스를 가집니다.
    return vocab

# 'tokenized_title'에서 단어 사전을 구축합니다.
vocab = build_vocab(data['tokenized_title'])

# 단어 사전(vocab)을 출력하여 단어와 해당 인덱스를 확인합니다.
print(f"vocab : {vocab}")


# ------------------------- 텍스트를 정수 인덱스 시퀀스로 변환 --------------------------------------------
# 각 단어를 고유한 정수 인덱스로 변환하여 시퀀스를 만듭니다. 모델에 입력으로 사용할 수 있게 고정된 길이로 변환합니다.
# 시퀀스가 최대 길이를 초과하면 잘라내고, 짧으면 0으로 패딩(빈 공간 채우기)하여 고정 길이로 맞춥니다.
def text_to_sequence(tokenized_text, vocab, max_len=20):
    # 단어 리스트를 사전에 있는 단어에 대응하는 정수 인덱스로 변환합니다.
    # vocab에 없는 단어는 0으로 처리합니다.
    # vocab.get(word, 0): 단어가 vocab에 있으면 그 단어의 인덱스를, 없으면 0을 반환합니다.
    sequence = [vocab.get(word, 0) for word in tokenized_text[:max_len]]
    
    # 시퀀스가 최대 길이보다 짧으면 0으로 패딩하여 길이를 맞춥니다.
    # 최대 길이보다 짧은 경우 (max_len - len(sequence))만큼 0을 추가해 고정 길이를 유지합니다.
    if len(sequence) < max_len:
        sequence += [0] * (max_len - len(sequence))
    
    # 정수 인덱스 시퀀스를 반환합니다.
    return sequence

# 'tokenized_title' 열의 각 행에 대해 text_to_sequence 함수를 직접 적용합니다.
sequences = []
for index, row in data['tokenized_title'].items():
    sequences.append(text_to_sequence(row, vocab))

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


# ------------------------- 시드 고정 --------------------------------------------
# 재현 가능한 결과를 얻기 위해 시드를 고정합니다.
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

# 시드를 42로 고정합니다. 필요에 따라 다른 값으로 변경할 수 있습니다.
set_seed(42)


# ------------------------- RNN 모델 정의 --------------------------------------------
# 텍스트 데이터를 처리할 RNN(Recurrent Neural Network) 모델을 정의합니다.
# 단어를 벡터로 변환한 후, RNN을 통해 순차적인 의존성을 학습하고, 마지막 hidden state를 기반으로 이진 분류를 수행합니다.
class TrafficRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, output_dim=1):
        super(TrafficRNN, self).__init__()
        # nn.Embedding: 단어를 고정된 크기의 임베딩 벡터로 변환합니다.
        # vocab_size는 단어 집합의 크기, embedding_dim은 임베딩 벡터의 차원입니다.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # nn.RNN: 순차적인 데이터를 처리하는 RNN 레이어입니다. hidden_dim은 RNN의 은닉 상태 차원입니다.
        # batch_first=True를 사용하면 입력 텐서의 첫 번째 차원이 배치 크기로 설정됩니다.
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        # nn.Linear: RNN의 마지막 hidden state를 입력으로 받아 최종 출력을 만드는 완전 연결 레이어입니다.
        # output_dim=1은 이진 분류를 위한 출력 노드 수입니다.
        self.fc = nn.Linear(hidden_dim, output_dim)
        # nn.Sigmoid: 출력 값을 0과 1 사이의 값으로 변환하여 이진 분류를 수행합니다.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 임베딩 레이어를 사용하여 입력 텍스트 시퀀스를 고정된 크기의 벡터로 변환합니다.
        embedded = self.embedding(x)
        # RNN 레이어를 통해 순차적인 데이터를 처리하고 hidden state를 얻습니다.
        # 여기서 output이 사용되지 않는 이유는 해당 태스크에서는 마지막 값으로만 판단하기 때문입니다.
        output, hidden = self.rnn(embedded)
        # RNN의 마지막 hidden state만을 사용하여 완전 연결층으로 전달합니다.
        hidden = hidden.squeeze(0)
        # 최종 출력을 sigmoid를 통해 0과 1 사이의 확률 값으로 변환합니다.
        out = self.fc(hidden)
        return self.sigmoid(out)

# ------------------------- 하이퍼파라미터 설정 및 모델 초기화 --------------------------------------------
# vocab_size는 단어 사전의 크기이며, 임베딩 및 RNN의 차원을 설정합니다.
# vocab_size에는 단어 집합의 크기를 사용하며, 모델을 초기화합니다.
vocab_size = len(vocab) + 1  # 단어 집합 크기 (단어가 없는 경우를 위한 +1)
model = TrafficRNN(vocab_size).to(device)  # 모델을 GPU/CPU로 설정

# ------------------------- 손실 함수 및 옵티마이저 설정 --------------------------------------------
# 이진 교차 엔트로피 손실 함수를 사용하여 모델의 예측과 실제 값 사이의 손실을 계산합니다.
criterion = nn.BCELoss()  # 이진 분류를 위한 손실 함수 (Binary Cross-Entropy Loss)
# Adam 옵티마이저는 가중치를 업데이트하기 위한 최적화 알고리즘입니다. 학습률(lr=0.001)을 설정합니다.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------- 모델 학습 루프 --------------------------------------------
# 지정된 에포크 수만큼 모델을 학습합니다. 학습 중에는 옵티마이저가 손실을 기반으로 모델의 가중치를 업데이트합니다.
num_epochs = 5  # 총 5번의 학습 에포크를 수행
for epoch in range(num_epochs):
    model.train()  # 모델을 학습 모드로 전환
    optimizer.zero_grad()  # 옵티마이저의 기울기를 초기화
    predictions = model(X_train.to(device)).squeeze()  # 모델 예측 수행
    # 모델의 예측과 실제 라벨(y_train) 간의 손실을 계산
    loss = criterion(predictions, y_train.float().to(device))
    loss.backward()  # 손실에 대한 기울기를 계산
    optimizer.step()  # 옵티마이저가 가중치를 업데이트
    # 현재 에포크에서의 손실 값을 출력
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')  # 에포크와 손실 출력
