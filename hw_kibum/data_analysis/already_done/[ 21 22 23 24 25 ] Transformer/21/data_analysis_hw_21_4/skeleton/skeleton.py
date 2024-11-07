import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------------------- NLTK 데이터 다운로드 --------------------------------------------
# TODO: NLTK의 'punkt' 데이터를 다운로드하세요.
# NLTK(Natural Language Toolkit)는 자연어 처리(NLP)를 위한 라이브러리입니다.
# 'punkt' 데이터는 단어 및 문장 토크나이징을 위해 필요합니다.
nltk.download('punkt')
nltk.download('punkt_tab')

# ------------------------- 데이터 준비 --------------------------------------------
# 교통 관련 텍스트 데이터를 예시로 사용
data = pd.DataFrame({
    '제목': ['교통사고 분석', '도시 교통량 예측', '교통사고 원인 분석', '고속도로 사고', '교통량 변화 분석', '사고 예방 방안'],
    '라벨': ['사고', '예측', '사고', '사고', '예측', '예방']
})

# ------------------------- 라벨 인코딩 --------------------------------------------
# 텍스트 라벨을 모델이 이해할 수 있는 숫자로 변환하기 위해 LabelEncoder를 사용합니다.
# 예: '사고'는 0, '예측'은 1, '예방'은 2로 변환됩니다.
# TODO: LabelEncoder를 사용해 텍스트 라벨을 숫자로 변환하세요.
label_encoder = LabelEncoder()
data['라벨'] = label_encoder.fit_transform(data['라벨'])


# ------------------------- 텍스트 전처리 함수 정의 --------------------------------------------
# 텍스트를 소문자로 변환하고, 알파벳과 숫자를 제외한 문자를 제거합니다.
def clean_text(text):
    text = text.lower()
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    return text

# 데이터의 '제목' 열에 대해 clean_text 함수를 적용하여 텍스트 전처리를 진행합니다.
# TODO: 텍스트 전처리 및 토큰화를 진행하세요.
data['cleaned_title'] = data['제목'].apply(clean_text)

# ------------------------- 토크나이징 --------------------------------------------
# NLTK의 word_tokenize 함수를 사용하여 텍스트를 단어 단위로 나눕니다.
# 이 과정은 머신러닝 모델에서 텍스트 데이터를 처리할 수 있는 형태로 만듭니다.
data['tokenized_title'] = data['cleaned_title'].apply(word_tokenize)

# ------------------------- 단어 집합(vocab) 생성 --------------------------------------------
# 각 단어에 대해 고유한 인덱스를 부여합니다. 데이터에 포함된 단어를 집합으로 만들고, 각 단어에 인덱스를 할당합니다.
# MAX_LEN은 문장의 최대 길이로, 짧은 문장은 패딩을 통해 길이를 맞춥니다.
MAX_LEN = 10

vocab = {word: idx + 1 for idx, word in enumerate(set(sum(data['tokenized_title'], [])))}  # 단어에 고유 인덱스 부여
vocab_size = len(vocab) + 1  # 패딩을 위한 0을 고려하여 vocab 크기를 1 증가시킴

# ------------------------- 텍스트를 시퀀스로 변환 --------------------------------------------
# 텍스트 데이터를 정수 인덱스 시퀀스로 변환하는 함수
def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, 0) for word in text]  # 해당하는 단어를 인덱스로 변환
    if len(sequence) < max_len:
        sequence += [0] * (max_len - len(sequence))  # 짧은 문장은 0으로 패딩
    return sequence[:max_len]  # 최대 길이를 초과하는 경우 자름

# 데이터의 'tokenized_title' 열에 대해 text_to_sequence 함수를 적용하여 시퀀스를 생성합니다.
# data['input_ids'] = data['tokenized_title'].apply(text_to_sequence)
data['input_ids'] = data['tokenized_title'].apply(lambda x: text_to_sequence(x, vocab, MAX_LEN))

# ------------------------- Dataset & DataLoader --------------------------------------------
# Dataset 클래스는 데이터셋을 관리합니다. 모델에 사용할 입력 시퀀스와 라벨을 반환합니다.
class TextDataset(Dataset):
    def __init__(self, data):
        self.input_ids = list(data['input_ids'])
        self.labels = list(data['라벨'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx]), torch.tensor(self.labels[idx])

# ------------------------- 데이터셋 분할 --------------------------------------------
# 데이터를 학습셋과 테스트셋으로 나누어 모델이 새로운 데이터에 대해 어떻게 예측하는지 평가합니다.
# TODO: 데이터를 학습(train)과 테스트(test)로 나누세요.
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 학습과 테스트 데이터에 대해 Dataset을 만듭니다.
train_dataset = TextDataset(train_data)
test_dataset = TextDataset(test_data)

# DataLoader를 사용하여 데이터를 배치(batch)로 나누어 모델에 전달합니다. 배치 크기는 2로 설정합니다.
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# ------------------------- Transformer 모델 정의 --------------------------------------------
# Positional Encoding: Transformer는 순서 정보를 학습하지 않기 때문에 위치 정보를 임베딩에 추가합니다.
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout_p=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        # 위치 인코딩 계산
        pos_encoding = torch.zeros(max_len, embedding_dim)
        positions_list = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0)) / embedding_dim)

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, token_embedding):
        seq_len = token_embedding.size(1)
        pos_encoding = self.pos_encoding[:, :seq_len, :]
        return self.dropout(token_embedding + pos_encoding)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_heads=4, hidden_dim=256, num_encoder_layers=2,
                 output_dim=3, dropout_p=0.1):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout_p)

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout=dropout_p)
            for _ in range(num_encoder_layers)
        ])

        self.fc = nn.Linear(embedding_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src):
        src = self.embedding(src)
        src = self.positional_encoding(src)

        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src)

        src = src.mean(dim=1)  # 평균 풀링
        output = self.fc(src)
        return self.softmax(output)

# ------------------------- 학습 및 평가 --------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(vocab_size=vocab_size, output_dim=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for input_ids, labels in train_loader:
        input_ids, labels = input_ids.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

# 평가 함수
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, labels in test_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            outputs = model(input_ids)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 모델 학습
for epoch in range(10):  # 10 에포크 학습
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    test_accuracy = evaluate_model(model, test_loader, device)
    print(f'Epoch {epoch+1}, Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
