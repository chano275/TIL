import math
import torch
import torch.nn as nn
'''
해당 과제를 통해 Transformer 구조를 코드로 이해하는 것을 목표로 합니다.
'''

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

# --------------------- 모듈 출력 ---------------------
'''
아래는 각각의 모듈이 올바르게 정의되었는지 확인하는 과정입니다.
Positional Encoding, Transformer Encoder Layer, Transformer Decoder Layer가 올바르게 출력되는지 확인합니다.
'''

# 1. Positional Encoding 모듈 출력
pos_encoding = PositionalEncoding(embedding_dim=128)
print("Positional Encoding Module:")
print(pos_encoding)

# 2. Transformer Encoder Layer 모듈 출력
encoder_layer = TransformerEncoderLayer(embedding_dim=128, num_heads=4, hidden_dim=256)
print("\nTransformer Encoder Layer Module:")
print(encoder_layer)

# 3. Transformer Decoder Layer 모듈 출력
decoder_layer = TransformerDecoderLayer(embedding_dim=128, num_heads=4, hidden_dim=256)
print("\nTransformer Decoder Layer Module:")
print(decoder_layer)
