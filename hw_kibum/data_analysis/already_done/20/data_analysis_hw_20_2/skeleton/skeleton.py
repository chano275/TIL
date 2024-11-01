import torch
import torch.nn as nn
import torch.optim as optim
from torch.onnx.symbolic_opset11 import unsqueeze

# 현재 데이터가 문자 => TEXT를 문자 => 숫자로 토크나이징 해줘야 한다.
# 토크나이징 : 토큰으로 쪼갠다 > 가장 간단한 방법 : 공백으로 쪼갠다 .

# 간단한 학습 데이터셋 - 영화 리뷰 문장과 긍정(1) 또는 부정(0) 레이블을 함께 포함하고 있습니다.
train_data = [    ("I love this movie", 1),    ("This film was terrible", 0),    ("Absolutely fantastic!", 1),    ("I did not like this movie", 0),]
# 간단한 테스트 데이터셋 - 학습된 모델이 테스트 데이터에서 얼마나 잘 동작하는지 평가합니다.
test_data = [    ("This movie is amazing", 1),    ("I hated the film", 0),]

# 단어별로 나누기 위한 함수 (토큰화) - 문장을 공백을 기준으로 단어로 나누어 리스트로 반환합니다.
# 예: "I love this movie" -> ["I", "love", "this", "movie"]
def tokenize(text):return text.split()


### 어휘집 생성: 학습 데이터에서 모든 단어를 모아 고유한 번호(인덱스)를 부여합니다.
vocab = {word for sentence, _ in train_data for word in tokenize(sentence)}
vocab = {word: idx for idx, word in enumerate(vocab, 1)}  # 단어에 인덱스 부여
# 결과: {"I": 1, "love": 2, "this": 3, "movie": 4, ...}


# 텍스트를 숫자로 변환하는 함수 - 각 단어를 어휘집에 따라 숫자로 바꿉니다. 어휘집에 없는 단어는 0으로 처리합니다.
# 예: "I love this movie" -> [1, 2, 3, 4]
def text_pipeline(sentence):return [vocab.get(word, 0) for word in tokenize(sentence)]

# 학습 데이터와 테스트 데이터를 숫자로 변환한 후, 텐서로 만듭니다. - 텐서는 파이토치에서 데이터를 처리하는 기본 형식입니다.
# 모델 INPUT은 무조건 텐서 [ 여기서 만들어놓은 데이터 & 라벨 = DATASET + DATA LOADER ]
train_texts = [torch.tensor(text_pipeline(sentence)) for sentence, _ in train_data]
train_labels = [torch.tensor(label) for _, label in train_data]
test_texts = [torch.tensor(text_pipeline(sentence)) for sentence, _ in test_data]
test_labels = [torch.tensor(label) for _, label in test_data]

# RNN 모델 정의
class RNN(nn.Module):  # 이미 추상화되어 있음
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        # 원 핫 인코딩 대신 사용함 ( word2vec > dense vector )
        # 임베딩하는 것 : 각각의 token들이 의미하는 특징 벡터를 만든다고 생각
        self.embedding = nn.Embedding(input_dim, embedding_dim)         # 단어를 고정된 크기의 벡터로 변환하는 임베딩 레이어입니다.
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)  # RNN 레이어: 단어 벡터를 순차적으로 입력받아 학습합니다. - 참고: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.fc = nn.Linear(hidden_dim, output_dim)                     # 완전 연결 레이어: RNN의 출력에서 최종 예측을 수행합니다.


    def forward(self, text):
        embedded = self.embedding(text)      # 입력된 텍스트를 임베딩 벡터로 변환합니다.
        output, hidden = self.rnn(embedded)  # RNN을 사용해 임베딩된 단어들을 처리합니다.
        return self.fc(hidden.squeeze(0))    # RNN의 출력값을 완전 연결 레이어를 통해 최종 예측값으로 변환합니다.


# 하이퍼파라미터 설정
INPUT_DIM = len(vocab) + 1  # 어휘 크기 + 패딩 인덱스
EMBEDDING_DIM = 100  # 임베딩 벡터 차원 (100차원)
HIDDEN_DIM = 128  # RNN 은닉 노드 수
OUTPUT_DIM = 2  # 출력 차원 (긍정/부정)

# 모델 생성
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

# 옵티마이저와 손실 함수 정의
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# 학습 함수 정의
def train(model, texts, labels, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for text, label in zip(texts, labels):
        optimizer.zero_grad()
        # 예측 생성을 위해 모델에 텍스트를 입력합니다.
        # unsqueeze(0)을 사용하여 텐서에 배치 차원을 추가합니다. (자세한 설명: https://pytorch.org/docs/stable/generated/torch.unsqueeze.html#torch.unsqueeze)
        # 이렇게 하면 모델이 기대하는 입력 형태로 변환됩니다.
        predictions = model(text.unsqueeze(0))

        # 손실 함수를 사용해 예측값과 실제 레이블 간의 차이를 계산합니다.
        # 첫 번째 인자는 모델의 예측값(predictions)입니다.
        # 두 번째 인자는 실제 레이블(label)로, CrossEntropyLoss를 사용하기 위해 정수형(long)으로 변환하고 unsqueeze(0)을 사용해 차원을 맞춥니다.
        # 자세한 설명: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss = criterion(predictions, label.unsqueeze(0).long())

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(texts)

# 평가 함수 정의
def evaluate(model, texts, labels):
    model.eval()
    correct = 0
    with torch.no_grad():
        for text, label in zip(texts, labels):
            prediction = model(text.unsqueeze(0))
            prediction = torch.argmax(prediction, dim=1)  # 가장 높은 확률을 가진 클래스를 예측
            correct += (prediction == label).sum().item()  # 예측이 맞으면 카운트 증가
    return correct / len(labels)

# 학습 및 평가 실행
EPOCHS = 5
for epoch in range(EPOCHS):
    train_loss = train(model, train_texts, train_labels, optimizer, criterion)
    test_accuracy = evaluate(model, test_texts, test_labels)
    print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}, Test Accuracy: {test_accuracy:.3f}')

"""
성능 불일치 설명

이 코드에서 loss는 줄어들지만, accuracy는 일관되지 않게 변할 수 있습니다.
그 이유는:
1. 작은 데이터셋 크기: 모델이 다양한 패턴을 학습하지 못합니다.
2. 가중치 초기화: RNN 가중치가 매번 무작위로 초기화되어 결과가 다를 수 있습니다.
3. 과적합: 모델이 훈련 데이터에 너무 맞춰져, 테스트 성능이 저하될 수 있습니다.

이를 해결하려면 데이터셋을 확장하거나 정규화 기법을 적용할 수 있습니다.
"""
