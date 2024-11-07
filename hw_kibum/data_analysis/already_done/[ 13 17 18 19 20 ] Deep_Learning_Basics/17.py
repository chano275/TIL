import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


"""
첫 번째 모델 학습 및 평가:

특징으로 ['8시', '9시', '10시']를 사용하여 모델을 학습하고 평가했습니다.
두 번째 모델 학습 및 평가 (확장 모델):

특징을 ['7시', '8시', '9시', '10시']로 확장하여 모델을 다시 학습하고 평가했습니다.
손글씨 숫자 데이터셋 예제 추가:

DigitNet 클래스를 정의하여 MNIST 손글씨 숫자 데이터셋을 학습하고 평가했습니다.
이 부분은 TrafficDataset과 직접적인 연관은 없지만, 코드의 일관성을 위해 하나의 파일에 포함했습니다.
"""

# 랜덤 시드 고정 함수
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 랜덤 시드 고정
set_seed(42)

# GPU/CPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 장치: {device}")


# TrafficDataset 클래스 정의
class TrafficDataset(Dataset):
    def __init__(self, excel_file, features=['8시', '9시', '10시'], label='혼잡', sheet_name=0):
        # 데이터 로드
        self.data = pd.read_excel(excel_file, sheet_name=sheet_name)
        # 라벨을 정수형으로 변환
        self.data[label] = self.data[label].astype(int)
        # 특징과 라벨 설정
        self.features = self.data[features]
        self.labels = self.data[label]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.features.iloc[idx].values.astype(float)
        sample_tensor = torch.tensor(sample_data, dtype=torch.float32)
        label_data = self.labels.iloc[idx]
        label_tensor = torch.tensor(label_data, dtype=torch.long)
        return sample_tensor, label_tensor


# TrafficMLP 모델 정의
class TrafficMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TrafficMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 메인 코드
if __name__ == '__main__':
    # 데이터셋 및 데이터로더 설정
    input_features = ['8시', '9시', '10시']
    dataset = TrafficDataset('../data/weekday_traffic.xlsx', features=input_features)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 모델, 손실 함수, 옵티마이저 설정
    input_size = len(input_features)
    model = TrafficMLP(input_size=input_size, hidden_size=64, output_size=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 모델 학습
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    # 모델 평가
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    accuracy = accuracy_score(actuals, predictions)
    print(f'모델 정확도: {accuracy * 100:.2f}%')

    # 추가: 다른 특징 사용 ('7시', '8시', '9시', '10시')
    input_features_extended = ['7시', '8시', '9시', '10시']
    dataset_extended = TrafficDataset('../data/weekday_traffic.xlsx', features=input_features_extended)
    dataloader_extended = DataLoader(dataset_extended, batch_size=32, shuffle=True)

    # 새로운 입력 크기에 맞게 모델 재정의
    input_size_extended = len(input_features_extended)
    model_extended = TrafficMLP(input_size=input_size_extended, hidden_size=64, output_size=2).to(device)
    optimizer_extended = optim.Adam(model_extended.parameters(), lr=0.001)

    # 확장된 모델 학습
    model_extended.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader_extended:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer_extended.zero_grad()
            outputs = model_extended(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_extended.step()

            running_loss += loss.item()
        print(f"확장 모델 Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader_extended):.4f}")

    # 확장된 모델 평가
    model_extended.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in dataloader_extended:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_extended(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    accuracy_extended = accuracy_score(actuals, predictions)
    print(f'확장 모델 정확도: {accuracy_extended * 100:.2f}%')

    # 추가: 손글씨 숫자 데이터셋 사용 예제
    # 데이터셋 로드
    digits = load_digits()
    X = torch.FloatTensor(digits.data)
    y = torch.LongTensor(digits.target)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    # 모델 정의
    class DigitNet(nn.Module):
        def __init__(self):
            super(DigitNet, self).__init__()
            self.fc1 = nn.Linear(64, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 10)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    digit_model = DigitNet().to(device)
    criterion_digit = nn.CrossEntropyLoss()
    optimizer_digit = optim.SGD(digit_model.parameters(), lr=0.01)

    # 모델 학습
    num_epochs_digit = 50
    digit_model.train()
    for epoch in range(num_epochs_digit):
        optimizer_digit.zero_grad()
        outputs = digit_model(X_train.to(device))
        loss = criterion_digit(outputs, y_train.to(device))
        loss.backward()
        optimizer_digit.step()

        if (epoch + 1) % 10 == 0:
            print(f'Digit Model Epoch [{epoch + 1}/{num_epochs_digit}], Loss: {loss.item():.4f}')

    # 모델 평가
    digit_model.eval()
    with torch.no_grad():
        outputs = digit_model(X_test.to(device))
        _, predicted = torch.max(outputs.data, 1)
        accuracy_digit = accuracy_score(y_test.cpu(), predicted.cpu())
        print(f'\nDigit Model Test Accuracy: {accuracy_digit * 100:.2f}%')
