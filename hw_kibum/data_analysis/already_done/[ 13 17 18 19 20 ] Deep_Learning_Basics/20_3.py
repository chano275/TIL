import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # tqdm 라이브러리 임포트

# GPU 또는 CPU 사용 설정
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') # Mac
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Windows
print(f'Using device: {device}')  # 현재 사용 중인 디바이스 출력

# 셰익스피어 텍스트 데이터 로드 및 전처리
# 텍스트 데이터를 읽어 고유한 문자 집합을 추출합니다.
text = open('../data/shakespeare.txt').read()
chars = sorted(list(set(text)))  # 텍스트에 등장하는 모든 고유 문자를 추출하여 정렬
n_chars = len(chars)  # 고유한 문자 수 계산

# 각 문자를 고유한 숫자 인덱스로 변환하기 위한 딕셔너리 생성
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

# 시퀀스 길이 설정 및 데이터 준비
seq_length = 100 # 모델에 입력할 시퀀스 길이(100개의 문자로 구성된 텍스트 조각)
dataX = []  # 입력 시퀀스
dataY = []  # 정답 레이블(다음 문자)

# 텍스트 데이터를 시퀀스와 레이블로 나누는 과정
# 100글자씩 나누어 입력 시퀀스를 만들고, 그 다음 오는 문자를 레이블로 설정합니다.
for i in range(0, len(text) - seq_length):
    seq_in = text[i:i + seq_length]  # 입력 시퀀스 (100글자)
    seq_out = text[i + seq_length]  # 해당 시퀀스의 다음 글자
    dataX.append([char_to_idx[char] for char in seq_in])  # 각 문자를 숫자로 변환한 시퀀스 저장
    dataY.append(char_to_idx[seq_out])  # 다음에 올 문자의 인덱스를 저장


# one-hot encoding 함수 정의
# 시퀀스 데이터를 원핫 인코딩하여 모델에 사용할 수 있는 형태로 변환합니다.
def one_hot_encode(sequence, n_labels):
    encoding = np.zeros((len(sequence), n_labels))  # 시퀀스의 길이와 라벨 수에 맞는 0행렬 생성
    for i, value in enumerate(sequence):
        encoding[i, value] = 1  # 해당 문자에 해당하는 위치에 1을 할당
    return encoding


# 하나의 문자를 one-hot encoding하는 함수
def one_hot_encode_char(char_idx, n_labels):
    encoding = np.zeros((1, n_labels), dtype=np.float32)  # 하나의 문자를 위한 0행렬 생성
    encoding[0, char_idx] = 1  # 해당 인덱스 위치에 1을 할당 (문자 -> one-hot encoding)
    return encoding


# RNN 모델 정의
# Recurrent Neural Network (RNN) 모델로 텍스트 데이터를 학습합니다.
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # RNN 레이어
        self.fc = nn.Linear(hidden_size, output_size)  # 완전 연결 레이어 정의 (출력 레이어)

    def forward(self, x, hidden):
        # RNN 모델의 순전파(Forward Propagation)를 정의합니다.
        out, hidden = self.rnn(x, hidden)  # RNN에서 입력 시퀀스를 처리하고 hidden state를 업데이트
        if len(out.shape) == 2:  # 배치 크기가 1인 경우
            out = out[-1, :]  # 마지막 타임스텝의 출력만을 가져옵니다.
        else:
            out = out[:, -1, :]  # 마지막 타임스텝에서 모든 배치에 대해 출력
        out = self.fc(out)  # 완전 연결 레이어에서 최종 출력 생성
        return out, hidden  # 최종 출력값과 hidden state 반환


# 하이퍼파라미터 설정
hidden_size = 512
n_layers = 1

# RNN 모델 생성
model = RNN(n_chars, hidden_size, n_chars).to(device)  # 모델을 GPU로 이동

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)  # SGD 옵티마이저

# one-hot encoding을 적용한 데이터 변환
X = np.array([one_hot_encode(seq, n_chars) for seq in dataX], dtype=np.float32)  # one-hot 인코딩 후 텐서로 변환
X = torch.tensor(X).to(device)  # 데이터를 GPU로 이동
Y = torch.tensor(dataY).to(device)  # 정답 레이블을 GPU로 이동

# 텐서 데이터셋 생성 (데이터와 레이블을 포함한 데이터셋)
dataset = TensorDataset(X, Y)

# DataLoader를 사용하여 배치 단위로 데이터를 불러옴
batch_size = 64
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 학습 루프
def train_model(model, train_loader, epochs=50):
    for epoch in range(epochs):
        model.train()  # 모델을 학습 모드로 설정
        total_loss = 0  # 에포크당 손실 값 누적

        # tqdm을 사용하여 프로그레스바 출력
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)

        for batch_X, batch_Y in progress_bar:
            hidden = None  # 초기 hidden state는 None으로 설정
            optimizer.zero_grad()  # 기울기 초기화

            # RNN 모델의 forward 부분에서 hidden state를 모델에 전달하여 데이터를 처리합니다.
            # 'model(batch_X, hidden)'을 사용하여 데이터를 입력하고 hidden state도 전달합니다.
            # 이때, batch_X는 입력 데이터, hidden은 이전 시퀀스 정보를 기억하는 역할을 합니다.
            output, hidden = model(batch_X, hidden)  # (힌트: model에 입력과 hidden state를 넣어줍니다.)

            # 손실 함수를 사용하여 모델이 예측한 결과와 실제 레이블 간의 차이를 계산합니다.
            # 'criterion(output, batch_Y)'를 사용하여 손실을 계산할 수 있습니다.
            # 'CrossEntropyLoss'는 분류 문제에서 자주 사용하는 손실 함수로, 모델의 예측값과 실제값 간의 차이를 계산해줍니다.
            loss = criterion(output, batch_Y)  # (힌트: 손실 함수를 사용하여 output과 Y의 차이를 계산합니다.)

            loss.backward()  # 역전파를 통해 기울기 계산
            optimizer.step()  # SGD 옵티마이저를 통해 가중치 업데이트

            total_loss += loss.item()  # 손실 값 누적

            # 현재 배치 손실값을 프로그레스바에 표시
            progress_bar.set_postfix(loss=loss.item())

        # 매 epoch마다 평균 손실 값 출력
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')


# 학습 실행
train_model(model, train_loader, epochs=3)


# 텍스트 생성 함수 정의
def generate_text(model, start_str, length=100):
    input_seq = torch.tensor(one_hot_encode_char(char_to_idx[start_str[0]], n_chars)).to(device)
    hidden = None  # 초기 hidden state는 None

    generated = start_str  # 시작 문자열 설정
    for _ in range(length):
        output, hidden = model(input_seq, hidden)  # 모델에 입력 데이터를 전달하여 다음 문자 예측
        top_idx = output.argmax(0).item()  # 가장 높은 확률을 가진 문자의 인덱스 선택
        generated += idx_to_char[top_idx]  # 예측된 문자를 생성된 텍스트에 추가
        input_seq = torch.tensor(one_hot_encode_char(top_idx, n_chars)).to(device)

    return generated  # 생성된 텍스트 반환

# 텍스트 생성 실행
start_string = "My name is"
print(generate_text(model, start_string))

"""
여러 번 실행해도 완벽한 문장을 생성하지 못한 이유:

현재 코드는 기본적인 RNN 모델을 사용하여 셰익스피어 텍스트를 학습하지만, 여러 번 실행해도 원하는 대로 완벽한 문장을 생성하지 못하는 경우가 많습니다. 그 이유는 다음과 같습니다:

1. 데이터셋의 크기 부족: 학습 데이터셋이 충분히 크지 않으면, 모델이 텍스트의 다양한 패턴을 학습하지 못해 문장을 제대로 생성하지 못합니다. 더 많은 데이터가 있으면 모델이 다양한 문맥을 학습할 수 있어 성능이 개선될 가능성이 큽니다.
2. 기본 RNN의 한계: RNN은 기본적으로 장기 의존성을 잘 처리하지 못하는 문제를 가지고 있습니다. 문장이 길어지면, RNN은 초반에 입력된 정보를 잊어버리기 쉽습니다. LSTM(Long Short-Term Memory)이나 GRU(Gated Recurrent Unit)과 같은 더 복잡한 순환 신경망 아키텍처를 사용하면 이 문제를 해결할 수 있습니다.
3. 에포크와 학습률: 학습 에포크 수가 적거나 학습률이 너무 크거나 작으면 모델이 적절한 성능을 내지 못할 수 있습니다. 에포크를 늘리거나, 학습률을 조정하는 것도 성능 개선에 도움이 됩니다.
4. 모델 크기: 현재 설정된 RNN의 은닉 노드 수(hidden size)가 모델 성능에 영향을 미칠 수 있습니다. 노드 수가 너무 적으면 모델이 충분히 복잡한 패턴을 학습하지 못하고, 너무 많으면 과적합이 발생할 가능성이 있습니다.

성능을 높이기 위한 방법:

1. 데이터셋 확장: 더 많은 텍스트 데이터를 수집하여 학습에 활용하면 모델이 다양한 패턴을 학습할 수 있어 성능이 개선됩니다.
2. LSTM이나 GRU 사용: RNN 대신 LSTM이나 GRU와 같은 더 복잡한 모델을 사용하면 장기 의존성 문제를 해결하고 성능을 개선할 수 있습니다.
3. 에포크 수와 학습률 조정: 더 많은 에포크로 모델을 학습시키고, 학습률을 적절하게 조정하면 성능 향상에 도움이 됩니다. 학습률이 너무 크면 최적의 가중치에 도달하지 못하고, 너무 작으면 학습이 느려질 수 있습니다.
4. 정규화 기법 적용: 드롭아웃(dropout)이나 정규화를 적용하여 과적합을 방지하고 모델의 일반화 성능을 향상시킬 수 있습니다.
"""
