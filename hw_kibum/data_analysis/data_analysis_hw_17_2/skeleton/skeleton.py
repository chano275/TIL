import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. 데이터셋 준비
digits = load_digits()
X = digits.data  # (n_samples, n_features)
y = digits.target  # (n_samples,)

# 텐서로 변환
X = torch.FloatTensor(X)
y = torch.LongTensor(y)

# 훈련 세트와 테스트 세트로 분할 (훈련 데이터 80%, 테스트 데이터 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. 모델 설계 및 구현
class DigitNet(nn.Module):
    def __init__(self):
        super(DigitNet, self).__init__()
        self.fc1 = nn.Linear(64, 32)  # 입력층 -> 은닉층
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)  # 은닉층 -> 출력층

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # 출력층 (로짓 출력)
        return x

model = DigitNet()

# 3. 모델 학습
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 옵티마이저 설정

num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)      # 순전파 - 모델에 입력값을 넣어 예측값을 계산합니다.
    loss = criterion(outputs, y_train)        # 손실 계산 - 손실 함수를 이용해 예측값과 실제값 사이의 차이를 계산합니다.
    loss.backward()                # 역전파 -  loss값으로부터 backword함수를 이용해 역전파를 수행합니다.
    optimizer.step()                # 가중치 업데이트 - 계산된 gradient를 이용해 가중치를 업데이트합니다.

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 4. 모델 평가
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = accuracy_score(y_test, predicted)
    print(f'\nTest Accuracy: {accuracy*100:.2f}%')
