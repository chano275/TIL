import math
import torch
import torch.nn as nn

# --------------------- Positional Encoding 클래스 ---------------------
'''
Transformer 모델은 순서 정보를 직접 학습할 수 없기 때문에,
입력 데이터에 위치 정보를 부여하여 문장 내 단어의 순서를 학습할 수 있도록 돕습니다.
PositionalEncoding 클래스는 각 단어 임베딩에 위치 정보를 추가하는 역할을 하며,
사인(Sin)과 코사인(Cosine) 함수를 사용하여 위치 정보를 계산합니다.
'''

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout_p=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        # Dropout 레이어로 과적합 방지
        self.dropout = nn.Dropout(p=dropout_p)

        # 위치 인코딩을 위한 사인 및 코사인 계산
        # pos_encoding의 역할: 각 단어 임베딩에 위치 정보를 더해주는 역할
        pos_encoding = torch.zeros(max_len, embedding_dim)
        # 위치 값 (문장 내에서 단어의 위치)
        positions_list = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 주기적인 패턴을 추가하기 위한 scaling factor 계산
        division_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0)) / embedding_dim)

        # 짝수 인덱스에 사인 함수를 적용하여 위치 인코딩 값을 계산
        # TODO: 짝수 인덱스에는 사인 함수를 적용하세요.
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        # 홀수 인덱스에 코사인 함수를 적용하여 위치 인코딩 값을 계산
        # TODO: 홀수 인덱스에는 코사인 함수를 적용하세요.
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # 배치 차원을 추가하여 각 시퀀스의 위치 인코딩을 독립적으로 적용할 수 있도록 합니다.
        pos_encoding = pos_encoding.unsqueeze(0)
        # 학습되지 않는 파라미터로 등록
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, token_embedding):
        # 입력 시퀀스 길이에 맞춰 위치 인코딩 값을 잘라 적용합니다.
        seq_len = token_embedding.size(1)
        pos_encoding = self.pos_encoding[:, :seq_len, :]
        # 입력 임베딩에 위치 인코딩을 더한 후 Dropout을 적용하여 과적합을 방지합니다.
        return self.dropout(token_embedding + pos_encoding)


# --------------------- Transformer Encoder Layer ---------------------
'''
Transformer 모델의 인코더 레이어는 Self-Attention과 Feed-Forward 신경망을 통해 
입력 문장 내 각 단어의 상호 관계를 학습합니다.
'''
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout_p=0.1):
        super(TransformerEncoderLayer, self).__init__()

        # Multihead Self-Attention 레이어. 각 단어 간의 관계를 학습합니다.
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_p)
        # Feed-Forward Network 레이어. 각 단어의 의미를 더욱 정교하게 학습합니다.
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4), # 임베딩 차원을 4배로 확장
            nn.ReLU(), # 비선형성을 추가하여 모델의 표현력을 강화
            nn.Dropout(dropout_p), # 과적합을 방지
            nn.Linear(embedding_dim * 4, embedding_dim) # 다시 원래 임베딩 차원으로 축소
        )
        # Layer Normalization 레이어. 입력값을 정규화하여 학습의 안정성을 높입니다.
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        # Dropout 레이어. 과적합을 방지합니다.
        self.dropout = nn.Dropout(dropout_p)
        # Self-Attention 값에 대한 스케일링을 위한 상수입니다.
        self.scale = math.sqrt(embedding_dim)

    def forward(self, x):
        # TODO: Layer Normalization을 적용한 Self-Attention을 수행하세요.
        x_norm = self.layer_norm1(x)
        attn_output, _ = self.self_attention(x_norm, x_norm, x_norm)
        # Residual Connection을 사용하여 Self-Attention 출력과 이전 층의 출력을 더합니다. 이는 학습 안정성을 높이고 기울기 소실 문제를 완화합니다.
        x = x + self.dropout(attn_output)

        # TODO: Layer Normalization을 적용한 후, Feed-Forward Network를 적용하세요.
        x_norm = self.layer_norm2(x)
        ff_output = self.feed_forward(x_norm)
        # Residual Connection을 사용하여 Feed-Forward Network의 출력과 이전 층의 출력을 더합니다. 이는 학습 안정성을 높이고 기울기 소실 문제를 완화합니다.
        x = x + self.dropout(ff_output)
        return x


# --------------------- Transformer Decoder Layer ---------------------
'''
디코더 레이어는 인코더에서 출력된 정보를 바탕으로 타겟 시퀀스를 예측하는 역할을 합니다.
Self-Attention과 Cross-Attention을 결합하여 입력 시퀀스와 타겟 시퀀스 간의 관계를 학습합니다.
'''
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout_p=0.1):
        super(TransformerDecoderLayer, self).__init__()

        # 디코더 Self-Attention: 타겟 시퀀스의 각 단어 간의 관계를 학습한다.
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_p)
        # Cross-Attention: 인코더의 출력과 디코더의 타겟 시퀀스 간의 관계를 학습한다.
        self.enc_dec_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_p)
        # Feed-Forward Network. 타겟 시퀀스의 각 단어를 정교하게 학습한다.
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

        # Layer Normalization 레이어. 학습의 안정성을 높이기 위해 입력을 정규화합니다.
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.layer_norm3 = nn.LayerNorm(embedding_dim)
        # Dropout 레이어. 과적합을 방지합니다.
        self.dropout = nn.Dropout(dropout_p)
        self.scale = math.sqrt(embedding_dim)

    def forward(self, x, memory, tgt_mask=None):
        # 디코더 Self-Attention 수행. 타겟 시퀀스의 각 단어 간 관계를 학습
        # TODO: Self-Attention을 수행한 후, Cross-Attention을 수행하세요.
        x_norm = self.layer_norm1(x)
        attn_output, _ = self.self_attention(x_norm / self.scale, x_norm, x_norm, attn_mask=tgt_mask)
        x = x + self.dropout(attn_output)

        # Cross-Attention 수행. 인코더의 출력(memory)과 디코더 타겟 시퀀스 간 관계 학습
        # TODO: Cross-Attention을 수행한 후, Feed-Forward Network를 적용하세요.
        # Cross-Attention 수행. 인코더의 출력(memory)과 디코더 타겟 시퀀스 간 관계 학습
        x_norm = self.layer_norm2(x)
        attn_output, _ = self.enc_dec_attention(x_norm / self.scale, memory, memory)
        x = x + self.dropout(attn_output)

        # Feed-Forward Network 적용
        ff_output = self.feed_forward(self.layer_norm3(x))
        x = x + self.dropout(ff_output)
        return x


# --------------------- Transformer Model 정의 ---------------------
'''
Transformer 모델은 주어진 입력 시퀀스에 대해 다중 클래스 분류를 수행하는 모델입니다.
Encoder와 Decoder 레이어를 결합하여 전체 모델 구조를 형성합니다.
'''
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_heads=4, hidden_dim=256, num_encoder_layers=6,
                 num_decoder_layers=6, output_dim=2, dropout_p=0.1):
        super(TransformerModel, self).__init__()

        # 단어를 벡터로 변환하는 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 단어 벡터에 위치 정보를 추가하는 Positional Encoding 레이어
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout_p)

        # 다수의 인코더 레이어를 쌓아 인코더를 구성
        # TODO: Encoder와 Decoder 레이어를 쌓으세요.
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_p) for _ in range(num_encoder_layers)])
        # 다수의 디코더 레이어를 쌓아 디코더를 구성
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_p) for _ in range(num_decoder_layers)])

        # 최종 출력 차원을 설정하는 Fully Connected Layer
        self.fc = nn.Linear(embedding_dim, output_dim)
        # Softmax 활성화 함수는 각 클래스에 대한 확률 분포를 생성하여 최종 예측값을 출력합니다.
        # 각 클래스의 확률 중 가장 높은 값을 예측 결과로 사용합니다.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 인코더 입력 (src)에 임베딩과 위치 인코딩을 적용
        # TODO: Embedding과 Positional Encoding을 적용하세요.
        # 디코더 입력 (tgt)에 임베딩과 위치 인코딩을 적용
        src = self.positional_encoding(self.embedding(src.long()))
        tgt = self.positional_encoding(self.embedding(tgt.long()))

        # 인코더를 순차적으로 적용
        memory = src
        for encoder in self.encoder_layers:
            memory = encoder(memory)

        # 디코더를 순차적으로 적용
        output = tgt
        for decoder in self.decoder_layers:
            output = decoder(output, memory, tgt_mask)

        # 평균 풀링을 적용한 후 Softmax로 최종 출력을 반환
        output = output.mean(dim=1)
        output = self.softmax(self.fc(output))
        return output

# --------------------- 모듈 실행 예시 ---------------------
# Transformer 모델 실행 예시 (vocab_size=10000 기준)
vocab_size = 10000
model = TransformerModel(vocab_size=vocab_size)

# 가짜 입력 데이터 생성 (배치 크기=32, 시퀀스 길이=20)
src = torch.randint(0, vocab_size, (32, 20))
tgt = torch.randint(0, vocab_size, (32, 20))

# 모델 실행
output = model(src, tgt)
print("Model output shape:", output.shape)
