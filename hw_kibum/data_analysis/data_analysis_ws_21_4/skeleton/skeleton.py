from collections import Counter
import re
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk
import math
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

# --------------------- Positional Encoding 클래스 ---------------------
'''
Transformer 모델은 순서 정보를 직접 학습할 수 없기 때문에,
입력 데이터에 위치 정보를 부여하여 문장 내 단어의 순서를 학습할 수 있도록 돕습니다.
PositionalEncoding 클래스는 각 단어 임베딩에 위치 정보를 추가하는 역할을 하며,
사인(Sin)과 코사인(Cosine) 함수를 사용하여 위치 정보를 계산합니다.
'''

# Positional Encoding 클래스: 문장의 각 단어가 문장에서의 위치를 인식할 수 있도록 위치 정보를 부여하는 역할
# Transformer는 순서 정보를 직접 인식할 수 없기 때문에, 각 단어에 위치 정보를 더해주는 역할
# 참고: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout_p=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # nn.Dropout: 입력 텐서의 일부 값을 0으로 설정하여 과적합을 방지하는 역할
        # 드롭아웃 확률은 dropout_p로 설정되며, 학습 과정에서 일부 뉴런을 무작위로 제거
        self.dropout = nn.Dropout(p=dropout_p)

        # torch.zeros: 모든 값이 0인 텐서를 생성합니다. max_len은 시퀀스의 최대 길이, embedding_dim은 임베딩 차원
        pos_encoding = torch.zeros(max_len, embedding_dim)
        
        # torch.arange: 0부터 max_len-1까지의 정수 시퀀스를 생성하여 위치 값을 나타냄
        positions_list = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # torch.exp: 지수 함수를 사용하여 division_term을 계산. 이는 위치 정보를 사인과 코사인에 적용할 때 스케일링하는 데 사용
        division_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0)) / embedding_dim)

        # TODO: 위치 정보를 사인과 코사인 함수를 사용해 부여 (짝수 인덱스에는 사인 torch.sin(), 홀수 인덱스에는 코사인 torch.cosine())
        pos_encoding[:, 0::2] = ___________________  # 짝수 인덱스에 대해 사인 함수 적용
        pos_encoding[:, 1::2] = ___________________    # 홀수 인덱스에 대해 코사인 함수 적용

        # 입력 시퀀스와 같은 형태로 배치 차원을 추가하여 텐서의 모양을 맞춤
        pos_encoding = pos_encoding.unsqueeze(0)

        # register_buffer: 학습되지 않는 파라미터로 저장, 모델의 상태에 포함되지만 학습 중에는 업데이트되지 않음
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, token_embedding):
        # token_embedding의 길이를 가져와 위치 인코딩을 적용할 범위를 지정
        seq_len = token_embedding.size(1)
        # 위치 인코딩을 해당 시퀀스 길이만큼 잘라서 적용
        pos_encoding = self.pos_encoding[:, :seq_len, :]
        
        # 위치 인코딩을 토큰 임베딩에 더한 뒤, 드롭아웃을 적용하여 최종 임베딩 반환
        return self.dropout(token_embedding + pos_encoding)

# --------------------- Transformer Encoder Layer ---------------------
'''
Transformer 모델의 인코더 레이어는 입력 문장 내 단어 간의 관계를 학습하는 데 사용됩니다.
Self-Attention(자기 어텐션)과 Feed-Forward Neural Network(FFN)를 결합하여
입력 시퀀스의 정보를 풍부하게 학습합니다.
'''

# Transformer 인코더 레이어: 주어진 입력에 대해 셀프 어텐션과 피드포워드 신경망을 사용하는 레이어
# 참고: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout_p=0.1):
        super(TransformerEncoderLayer, self).__init__()

        # nn.MultiheadAttention: Self-Attention을 여러 개의 헤드로 나누어 병렬로 처리.
        # embedding_dim: 입력 텐서의 차원, num_heads: 어텐션 헤드의 개수
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_p)

        # nn.Linear: 입력 텐서의 크기를 변환하는 선형 변환(fully connected layer)
        # 첫 번째 nn.Linear는 차원을 확장 (임베딩 차원의 4배로), 두 번째 nn.Linear는 다시 축소
        # 참고: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # 임베딩 차원의 4배로 확장
            nn.ReLU(),  # 활성화 함수 ReLU (음수는 0으로 변환)
            nn.Dropout(dropout_p),  # 과적합 방지를 위한 드롭아웃
            nn.Linear(embedding_dim * 4, embedding_dim)  # 다시 임베딩 차원으로 축소
        )

        # nn.LayerNorm: 입력 텐서의 평균과 분산을 정규화하여 학습의 안정성을 높임
        # 참고: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        # nn.Dropout: 드롭아웃을 사용해 과적합을 방지
        self.dropout = nn.Dropout(dropout_p)

        # Self-Attention 스케일링을 위한 상수
        self.scale = math.sqrt(embedding_dim)

    def forward(self, x):
        # 입력에 Layer Normalization을 먼저 적용한 후, Self-Attention을 수행
        x_norm = self.layer_norm1(x)

        # self_attention 함수는 쿼리, 키, 값의 입력을 받아 각 토큰 간의 관계를 계산
        # TODO: x_norm을 활용해서 주어진 빈칸에 값을 넣어주세요.
        attn_output, _ = self.self_attention(___________________, ___________________, ___________________)

        # Self-Attention 결과에 잔차 연결 (Residual Connection)과 드롭아웃을 적용하여 더함
        x = x + self.dropout(attn_output)

        # Self-Attention 이후, 두 번째 Layer Normalization을 적용한 후 FFN(Feed Forward Network)을 적용
        x_norm = self.layer_norm2(x)
        ff_output = self.feed_forward(x_norm)

        # FFN 결과에 다시 잔차 연결과 드롭아웃 적용
        x = x + self.dropout(ff_output)

        return x

# --------------------- Transformer Decoder Layer ---------------------
'''
디코더 레이어는 인코더에서 나온 출력과 타겟 문장(입력)을 사용하여
예측 값을 생성하는 역할을 합니다. 
Self-Attention과 Cross-Attention을 결합하여, 입력과 출력 시퀀스 간의 관계를 학습합니다.
'''

# Transformer 디코더 레이어: 인코더의 출력을 받아 디코더가 입력을 처리하는 레이어
# 참고: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout_p=0.1):
        super(TransformerDecoderLayer, self).__init__()

        # 디코더의 Self-Attention: 현재 디코더의 입력 토큰들 간의 관계를 학습
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_p)

        # 인코더 출력과 디코더 입력 간의 Cross-Attention: 인코더에서 나온 정보를 사용해 타겟 시퀀스를 예측
        self.enc_dec_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_p)

        # 디코더의 FFN도 인코더와 동일하게 임베딩 차원의 4배로 확장 후 다시 축소
        # FFN의 구조를 직접 구현하면서 차원을 확장하고 다시 줄이는 과정
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # 임베딩 차원의 4배로 확장
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(embedding_dim * 4, embedding_dim)  # 다시 축소
        )

        # 디코더에도 Layer Normalization을 적용
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.layer_norm3 = nn.LayerNorm(embedding_dim)

        # 드롭아웃 적용
        self.dropout = nn.Dropout(dropout_p)
        self.scale = math.sqrt(embedding_dim)

    def forward(self, x, memory, tgt_mask=None):
        # Self-Attention 적용: 디코더에서 입력된 시퀀스 간의 관계를 학습
        x_norm = self.layer_norm1(x)
        attn_output, _ = self.self_attention(x_norm / self.scale, x_norm, x_norm, attn_mask=tgt_mask)
        x = x + self.dropout(attn_output)

        # Cross-Attention 적용: 인코더 출력(memory)와 디코더 입력 간의 관계 학습
        # TODO: Cross-Attention을 수행하는 부분에서 인코더의 메모리(memory)와 디코더의 입력을 작성해주세요.
        x_norm = self.layer_norm2(x)
        attn_output, _ = self.enc_dec_attention(___________________, ___________________, ___________________)
        x = x + self.dropout(attn_output)

        # FFN 적용 후 잔차 연결 및 드롭아웃 적용
        x_norm = self.layer_norm3(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout(ff_output)

        return x

# --------------------- Transformer 모델 클래스 ---------------------
'''
Transformer 모델은 입력 시퀀스에 대해 다중 클래스 분류 작업을 수행하는 구조입니다. 
여기서 설명하는 클래스는 어휘(vocab_size) 크기, 임베딩 차원, 헤드 수, 레이어 수 등을 파라미터로 받아
문장을 임베딩하여 분류 작업을 수행합니다.
'''

# Transformer 모델 클래스: 주어진 입력 시퀀스에 대해 다중 클래스 분류를 수행하는 Transformer 모델
# 참고: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_heads=4, hidden_dim=256, num_encoder_layers=6, num_decoder_layers=6, output_dim=2, dropout_p=0.1):
        super(TransformerModel, self).__init__()
        
        # --------------------- 임베딩 레이어 (Embedding Layer) ---------------------
        '''
        단어를 고정된 차원(embedding_dim)의 벡터로 변환하여 처리할 수 있도록 변환합니다.
        임베딩을 통해 각 단어는 학습 가능한 고정 크기의 벡터로 매핑됩니다.
        '''
        # Embedding Layer: 단어를 고정된 차원(embedding_dim)의 벡터로 변환하는 역할
        # vocab_size: 어휘의 크기(단어 집합의 크기)
        # 참고: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # TODO: 빈칸에 들어갈 인자를 넣어주세요.
        self.embedding = nn.Embedding(___________, ___________)

        # --------------------- 포지셔널 인코딩 레이어 (Positional Encoding) ---------------------
        '''
        Transformer는 문장 내에서 단어 순서를 직접 학습하지 못하므로, 각 단어에 위치 정보를 부여하여
        단어 간의 상대적 위치를 학습할 수 있도록 돕습니다. 
        '''
        # Positional Encoding Layer: 각 단어의 위치 정보를 부여하여 문맥을 학습할 수 있도록 함
        # 참고: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        # TODO: 빈칸에 들어갈 인자를 넣어주세요.
        self.positional_encoding = PositionalEncoding(__________, __________, max_len=50)


        # --------------------- 인코더 레이어 리스트 (Encoder Layers) ---------------------
        '''
        Transformer의 인코더는 Self-Attention 메커니즘을 사용하여 입력된 시퀀스 내 단어 간의 관계를 학습합니다.
        각 인코더 레이어는 입력 시퀀스에서 정보를 처리하여 학습된 정보를 다음 레이어로 전달합니다.
        '''
        # Transformer 인코더 레이어를 쌓는 리스트. 각 인코더 레이어는 Self-Attention과 FFN을 포함
        # num_encoder_layers: 인코더 레이어의 개수
        # 참고: https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
        # TODO: 빈칸에 들어갈 인자를 넣어주세요.
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(__________, __________, __________, __________) for _ in range(num_encoder_layers)])

        # --------------------- 디코더 레이어 리스트 (Decoder Layers) ---------------------
        '''
        디코더 레이어는 인코더에서 처리된 정보를 바탕으로, 주어진 타겟 시퀀스와 함께 Self-Attention 및 Cross-Attention을 사용하여
        출력 시퀀스를 생성하는 역할을 합니다.
        '''
        # Transformer 디코더 레이어를 쌓는 리스트. 각 디코더 레이어는 Self-Attention, Cross-Attention, FFN을 포함
        # num_decoder_layers: 디코더 레이어의 개수
        # TODO: 빈칸에 들어갈 인자를 넣어주세요.
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(__________, __________, __________, __________) for _ in range(num_decoder_layers)])

        # --------------------- Fully Connected Layer (FC) ---------------------
        '''
        Transformer의 마지막 단계에서 나온 벡터를 분류 작업에 사용할 수 있도록 변환하는 완전 연결층(FC)입니다.
        여기서는 다중 클래스 분류를 수행하기 위해 output_dim으로 출력 차원을 설정합니다.
        '''
        # Fully Connected Layer (FC): 최종적으로 Transformer에서 나온 출력을 분류하기 위한 완전 연결층
        # 참고: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc = nn.Linear(embedding_dim, output_dim)  # 다중 클래스 분류를 위한 출력 차원

        # --------------------- Softmax 활성화 함수 ---------------------
        '''
        다중 클래스 분류에서는 출력층에서 Softmax 함수를 사용해 각 클래스에 대한 확률을 계산합니다.
        '''
        # Softmax 활성화 함수: 다중 클래스 분류에서 각 클래스의 확률을 출력
        # 참고: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.softmax = nn.Softmax(dim=-1)

    # --------------------- 순전파 (Forward) 과정 정의 ---------------------
    '''
    Transformer 모델의 순전파 과정은 입력된 시퀀스를 인코더로 처리한 후, 
    디코더를 통해 출력 시퀀스를 생성하고, 최종적으로 완전 연결층을 거쳐 Softmax로 확률 값을 반환합니다.
    '''
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src: 인코더 입력 (원본 문장), tgt: 디코더 입력 (타겟 문장)
        # src_mask, tgt_mask: 패딩 마스크 등을 적용할 수 있음 (옵션)

        # 1. 인코더와 디코더에서 임베딩과 포지셔널 인코딩 적용
        # 입력된 src (원본 문장)를 임베딩하고, 위치 인코딩을 적용
        src = self.embedding(src)  # 원본 문장에 대한 임베딩 적용
        src = self.positional_encoding(src)  # 위치 정보를 포함한 임베딩

        # 타겟 문장에도 동일하게 임베딩 및 포지셔널 인코딩을 적용
        tgt = self.embedding(tgt)  # 타겟 문장에 대한 임베딩 적용
        tgt = self.positional_encoding(tgt)  # 위치 정보를 포함한 임베딩

        # 2. 인코더 레이어를 순차적으로 적용하여 인코더 출력을 생성 (memory)
        memory = src  # 인코더의 입력을 memory에 저장
        for encoder in self.encoder_layers:
            memory = encoder(memory)  # 각 인코더 레이어가 memory를 갱신

        # 3. 디코더 레이어를 순차적으로 적용하여 타겟 시퀀스에 대한 출력을 생성
        output = tgt  # 디코더의 입력으로 타겟 문장을 사용
        for decoder in self.decoder_layers:
            output = decoder(output, memory, tgt_mask)  # 인코더의 memory를 사용해 디코더 출력 생성

        # 4. 평균 풀링 (Average Pooling): 전체 시퀀스의 출력을 평균내어 하나의 벡터로 변환
        # 각 시퀀스의 출력 벡터를 평균을 내어 하나의 고정된 크기의 벡터로 변환
        output = output.mean(dim=1)

        # 5. Fully Connected Layer와 Softmax 활성화 함수 적용
        # 완전 연결층을 통해 다중 클래스 분류를 위한 출력 차원으로 변환
        output = self.softmax(self.fc(output))  # 최종 출력을 분류 확률로 변환

        return output  # 다중 클래스 분류 결과 (각 클래스에 대한 확률)


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

# 단어 사전(vocab)을 출력하여 단어와 해당 인덱스를 확인합니다.
print(f"vocab : {vocab}")


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

# ------------------------- 텐서 변환 및 데이터 분할 --------------------------------------------
# 데이터를 PyTorch에서 사용할 수 있도록 텐서로 변환합니다. 텍스트 시퀀스(X)와 라벨(y)을 텐서로 변환하여 학습에 사용할 수 있게 만듭니다.
# torch.tensor: 리스트나 배열을 PyTorch 텐서로 변환합니다. 모델 학습에서 사용할 수 있도록 시퀀스 데이터를 텐서로 만듭니다.
X = torch.tensor(data['text_sequence'].tolist())  # 각 시퀀스를 텐서로 변환
y = torch.tensor(data['교통관련'].values)  # 라벨 데이터를 텐서로 변환

# 데이터를 훈련셋과 테스트셋으로 분할합니다. 이 과정은 모델이 새로운 데이터를 평가할 수 있도록 도와줍니다.
# train_test_split 함수는 데이터를 80:20 비율로 나누어 훈련셋과 테스트셋을 생성합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# vocab_size: 사전 크기 +1 (0 패딩 값 포함)
vocab_size = len(vocab) + 1

# Transformer 모델 초기화, 모델을 GPU나 CPU로 이동시킴
# TransformerModel은 앞서 정의된 클래스입니다.
model = TransformerModel(vocab_size=vocab_size).to(device)

# 모델의 구조를 출력하여 확인
print(f"model: {model}")