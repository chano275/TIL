import os, sys, torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader



if torch.cuda.is_available():  # GPU 사용 가능 여부 확인 / 참고 페이지: https://pytorch.org/docs/stable/cuda.html#torch.cuda.is_available
    # CUDA가 지원되는 GPU가 있는지 확인 (True/False 반환) / 참고 페이지: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
    # GPU 장치를 사용할 수 있으면 'cuda'로 설정 / 사용 가능한 첫 번째 GPU 장치의 이름 출력 / 참고 페이지: https://pytorch.org/docs/stable/cuda.html#torch.cuda.get_device_name
    device = torch.device('cuda')  
    print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
else:
    # CUDA가 없으면 'cpu'로 설정 / 참고 페이지: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
    device = torch.device('cpu')  
    print("GPU 사용 불가, CPU 사용")

#####


# Dataset 클래스 : Torch에서 데이터셋을 정의할 때 사용하는 기본 클래스
# __len__() : 데이터셋의 길이  /  __getitem__()  : 특정 인덱스의 데이터를 반환하는 방식으로 데이터셋을 정의
# PyTorch의 DataLoader와 함께 사용하여 데이터를 쉽게 배치 단위로 처리할 수 있습니다.
# 참고 페이지: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
# Dataset을 상속받아 교통데이터에 맞게 사용할 수 있도록 수정합니다.

class TrafficDataset(Dataset):
    def __init__(self, excel_file, sheet_name=0):
        self.data = pd.read_excel(excel_file, sheet_name=sheet_name)        # 데이터를 로드 (엑셀 파일을 읽어와 pandas DataFrame으로 저장)         
        self.data['혼잡'] = self.data['혼잡'].astype(int)        # 범주형 데이터를 숫자형으로 변환 (혼잡 여부: 0 또는 1로 변환)
        self.features = self.data[['8시', '9시', '10시']]        # 필요한 열만 선택하여 features에 저장 (8시, 9시, 10시 시간대의 교통량)       
        self.labels = self.data['혼잡']        # 혼잡 여부를 라벨로 설정 (이 데이터셋에서 타겟 값으로 사용)
        
    def __len__(self):        # 데이터셋의 길이 반환 (전체 샘플 개수)
        return len(self.data)
    
    def __getitem__(self, idx):
        sample_data = self.features.iloc[idx].values  # 특징 데이터를 가져옴 (NumPy 배열)
        sample_data = sample_data.astype(float)        # 2. 특징 데이터를 float 형으로 변환
        sample_tensor = torch.tensor(sample_data, dtype=torch.float32)        # 3. 특징 데이터를 PyTorch 텐서로 변환 / 참고: https://pytorch.org/docs/stable/generated/torch.tensor.html
        label_data = self.labels.iloc[idx]        # 4. 라벨(label)을 가져옴
        label_tensor = torch.tensor(label_data, dtype=torch.long)        # 5. 라벨 데이터를 PyTorch 텐서로 변환 / 참고: https://pytorch.org/docs/stable/generated/torch.tensor.html
        return sample_tensor, label_tensor        # 6. 텐서 형태의 특징과 라벨을 반환


traffic_dataset = TrafficDataset('../data/weekday_traffic.xlsx')    # 데이터셋 예시 사용
print(f"데이터셋 개수: {len(traffic_dataset)}")

# 데이터셋에서 특정 인덱스의 데이터를 가져오기
sample_data, sample_label = traffic_dataset[0]  # 첫 번째 데이터
print(f"첫 번째 데이터 (특징): {sample_data}")
print(f"첫 번째 데이터 (레이블): {sample_label}")
sample_data, sample_label = traffic_dataset[5]  # 두 번째 데이터
print(f"두 번째 데이터 (특징): {sample_data}")
print(f"두 번째 데이터 (레이블): {sample_label}")


# DataLoader 설정 - 데이터셋을 배치 단위로 로드하도록 돕는 클래스 / # 참고: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
# batch_size는 8로 설정 / shuffle=True는 데이터를 무작위로 섞어서 학습 / num_workers는 데이터를 로드하는 동안 병렬로 처리할 수 있도록 하는 프로세스 수를 지정 (기본값 0)
batch_size = 8
traffic_loader = DataLoader(traffic_dataset, batch_size=batch_size, shuffle=True)

# DataLoader를 통해 데이터 불러오기 - 배치 단위로 데이터를 불러오며, 이 예시에서는 각 배치의 특징 데이터와 라벨 데이터를 출력
for batch_idx, (data, target) in enumerate(traffic_loader):
    print(f"배치 {batch_idx+1} - 데이터 크기: {data.size()}, 레이블 크기: {target.size()}")      # 각 배치에서 가져온 데이터와 라벨의 크기를 출력
    
#####

import torch
import torch.nn as nn
import torch.nn.functional as F

# nn.Module을 상속받아 MLP 모델을 정의 (Multi-Layer Perceptron)
# nn.Module은 모든 신경망의 기본 클래스입니다. 신경망 레이어와 학습 가능한 파라미터를 관리하고, 순전파 연산을 정의하는 데 사용됩니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
class TrafficMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # nn.Module의 생성자를 호출하여 부모 클래스 초기화
        super(TrafficMLP, self).__init__()
        
        # 완전 연결층(fully connected layer) 정의
        # input_size에서 hidden_size로 가는 선형 변환 (가중치 및 편향 포함)
        self.fc1 = nn.Linear(input_size, hidden_size)  
        
        # 두 번째 완전 연결층 정의: hidden_size에서 output_size로 가는 선형 변환
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    # 순전파 연산 정의: 입력을 받아 레이어를 통과시키는 연산을 정의합니다.
    def forward(self, x):
        # 첫 번째 완전 연결층(fc1)을 통과한 후 ReLU 활성화 함수 적용
        # ReLU는 비선형 활성화 함수로, 음수 값을 0으로 변환하고 양수는 그대로 반환합니다.
        x = F.relu(self.fc1(x))  # 참고: https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
        
        # 두 번째 완전 연결층(fc2)을 통과한 후 출력
        # 이 레이어에서는 출력값이 그대로 반환되며, 보통 활성화 함수는 여기서는 사용되지 않음 (로짓값 출력)
        x = self.fc2(x)
        
        return x  # 최종 출력


if __name__ == '__main__':
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 인스턴스 생성
    # input_size는 입력 특성의 개수 (예: 8시, 9시, 10시 교통량의 3개 값)
    # hidden_size는 은닉층의 뉴런 개수 (64로 설정)
    # output_size는 분류할 클래스 개수 (혼잡/비혼잡, 2개 클래스)
    input_size = 3  # 예: 3개의 입력 특징 (8시, 9시, 10시 교통량)
    hidden_size = 64  # 은닉층 뉴런 개수
    output_size = 2  # 출력 클래스 개수 (혼잡/비혼잡 2개 클래스)

    # 모델을 인스턴스화하고, 'cuda'가 가능하면 GPU에, 아니면 CPU에 할당합니다.
    # 참고: https://pytorch.org/docs/stable/generated/torch.Tensor.to.html
    model = TrafficMLP(input_size, hidden_size, output_size).to(device)

    # 생성된 모델의 구조를 출력
    # 이 때 모델의 각 레이어 (입력 크기, 은닉층 크기, 출력 크기)와 레이어 정보가 출력됩니다.
    # 참고: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    print(model)
    
    
##### 

# 맨 마지막에 풀어주세요!!!!!!!!!!!!!!
import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import random
import numpy as np

# 랜덤 시드 고정 함수
# 학습의 일관성 및 재현성을 위해 모든 랜덤 시드를 고정합니다.
def set_seed(seed=42):
    random.seed(seed)  # Python의 기본 random 모듈 시드 설정
    np.random.seed(seed)  # Numpy 시드 설정
    torch.manual_seed(seed)  # PyTorch CPU 시드 설정
    if torch.cuda.is_available():  # GPU 사용 시 모든 GPU의 시드 설정
        torch.cuda.manual_seed_all(seed)

# 랜덤 시드 고정
set_seed(42)

"""
TODO:
practice_2에서 구현한 TrafficMLP Class 구현 전체와 practice_4에서 구현한 TrafficDataset Class 구현 전체를 붙여넣으세요.
"""
# Dataset 클래스는 PyTorch에서 데이터셋을 정의할 때 사용하는 기본 클래스입니다.
# __len__() 메서드를 구현하여 데이터셋의 길이를 반환하고,
# __getitem__() 메서드를 구현하여 특정 인덱스의 데이터를 반환하는 방식으로 데이터셋을 정의할 수 있습니다.
# PyTorch의 DataLoader와 함께 사용하여 데이터를 쉽게 배치 단위로 처리할 수 있습니다.
# 참고 페이지: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
# Dataset을 상속받아 교통데이터에 맞게 사용할 수 있도록 수정합니다.
class TrafficDataset(Dataset):
    def __init__(self, excel_file, sheet_name=0):
        # 데이터를 로드 (엑셀 파일을 읽어와 pandas DataFrame으로 저장)
        self.data = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # 범주형 데이터를 숫자형으로 변환 (혼잡 여부: 0 또는 1로 변환)
        self.data['혼잡'] = self.data['혼잡'].astype(int)

        # 필요한 열만 선택하여 features에 저장 (8시, 9시, 10시 시간대의 교통량)
        self.features = self.data[['8시', '9시', '10시']]
        
        # 혼잡 여부를 라벨로 설정 (이 데이터셋에서 타겟 값으로 사용)
        self.labels = self.data['혼잡']
        
    def __len__(self):
        # 데이터셋의 길이 반환 (전체 샘플 개수)
        return len(self.data)
    
    def __getitem__(self, idx):
        # 1. 특징(feature) 데이터를 가져옴
        sample_data = self.features.iloc[idx].values  # 특징 데이터를 가져옴 (NumPy 배열)
        
        # 2. 특징 데이터를 float 형으로 변환
        sample_data = sample_data.astype(float)
        
        # 3. 특징 데이터를 PyTorch 텐서로 변환
        # 참고: https://pytorch.org/docs/stable/generated/torch.tensor.html
        sample_tensor = torch.tensor(sample_data, dtype=torch.float32)
        
        # 4. 라벨(label)을 가져옴
        label_data = self.labels.iloc[idx]
        
        # 5. 라벨 데이터를 PyTorch 텐서로 변환
        # 참고: https://pytorch.org/docs/stable/generated/torch.tensor.html
        label_tensor = torch.tensor(label_data, dtype=torch.long)
        
        # 6. 텐서 형태의 특징과 라벨을 반환
        return sample_tensor, label_tensor

# nn.Module을 상속받아 MLP 모델을 정의 (Multi-Layer Perceptron)
# nn.Module은 모든 신경망의 기본 클래스입니다. 신경망 레이어와 학습 가능한 파라미터를 관리하고, 순전파 연산을 정의하는 데 사용됩니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
class TrafficMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # nn.Module의 생성자를 호출하여 부모 클래스 초기화
        super(TrafficMLP, self).__init__()
        
        # 완전 연결층(fully connected layer) 정의
        # input_size에서 hidden_size로 가는 선형 변환 (가중치 및 편향 포함)
        self.fc1 = nn.Linear(input_size, hidden_size)  
        
        # 두 번째 완전 연결층 정의: hidden_size에서 output_size로 가는 선형 변환
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    # 순전파 연산 정의: 입력을 받아 레이어를 통과시키는 연산을 정의합니다.
    def forward(self, x):
        # 첫 번째 완전 연결층(fc1)을 통과한 후 ReLU 활성화 함수 적용
        # ReLU는 비선형 활성화 함수로, 음수 값을 0으로 변환하고 양수는 그대로 반환합니다.
        x = F.relu(self.fc1(x))  # 참고: https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
        
        # 두 번째 완전 연결층(fc2)을 통과한 후 출력
        # 이 레이어에서는 출력값이 그대로 반환되며, 보통 활성화 함수는 여기서는 사용되지 않음 (로짓값 출력)
        x = self.fc2(x)
        
        return x  # 최종 출력


# GPU/CPU 설정
# GPU가 사용 가능하면 'cuda', 아니면 'cpu'로 모델 학습 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 및 데이터셋 설정
# TrafficMLP 모델 인스턴스를 생성하고 GPU/CPU로 전송
model = TrafficMLP(input_size=3, hidden_size=64, output_size=2).to(device)  
# TrafficDataset을 사용하여 엑셀 데이터를 불러오고 DataLoader에 적용
traffic_dataset = TrafficDataset('../data/weekday_traffic.xlsx')
# DataLoader는 데이터 배치를 관리하며, 학습 시 데이터를 섞어서 사용
traffic_loader = DataLoader(traffic_dataset, batch_size=32, shuffle=True)

# 손실 함수 및 옵티마이저 설정
# CrossEntropyLoss는 분류 문제에서 자주 사용되는 손실 함수입니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
criterion = nn.CrossEntropyLoss()  

# Adam 옵티마이저는 가중치를 업데이트할 때 사용하는 방법 중 하나입니다.
# 참고: https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 간단한 학습 루프 (10 에포크를 기준으로 진행해봅니다.)
# model.train()은 모델을 학습 모드로 전환합니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train
model.train()  # 학습 모드 설정

for epoch in range(10):  # 총 10 에포크 동안 학습
    running_loss = 0.0  # 에포크 동안의 손실을 누적할 변수
    for inputs, labels in traffic_loader:  # DataLoader를 통해 배치 단위로 데이터 불러오기
        inputs, labels = inputs.to(device), labels.to(device)  # 데이터를 GPU/CPU로 전송
        
        optimizer.zero_grad()  # 이전 배치에서 계산된 기울기 초기화
        outputs = model(inputs)  # 모델을 통해 예측값 생성 (순전파)
        loss = criterion(outputs, labels)  # 예측값과 실제값을 비교하여 손실 계산
        loss.backward()  # 손실에 따른 가중치의 기울기를 계산 (역전파)
        optimizer.step() # 계산된 기울기를 바탕으로 가중치 업데이트
        
        running_loss += loss.item()  # 손실을 누적
    # 에포크마다 평균 손실 출력
    print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(traffic_loader):.4f}")

# 평가 (모델 성능 측정)
# model.eval()은 모델을 평가 모드로 전환합니다. 드롭아웃과 배치 정규화가 비활성화됩니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval
model.eval()  # 평가 모드 설정
predictions = []  # 예측값 저장 리스트
actuals = []  # 실제값 저장 리스트

# 평가 모드에서는 기울기 계산이 필요 없으므로 no_grad()를 사용
# 참고: https://pytorch.org/docs/stable/generated/torch.no_grad.html
with torch.no_grad():  
    for inputs, labels in traffic_loader:
        inputs, labels = inputs.to(device), labels.to(device) # 데이터를 GPU/CPU로 전송
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # 예측된 클래스 선택 (참고: https://pytorch.org/docs/stable/generated/torch.max.html#torch-max)
        predictions.extend(predicted.cpu().numpy())  # 예측값 저장
        actuals.extend(labels.cpu().numpy())  # 실제값 저장

# sklearn의 accuracy_score를 사용하여 정확도 계산
accuracy = accuracy_score(actuals, predictions) * 100  # 정확도 계산

# 최종 정확도 출력
print(f'Accuracy: {accuracy:.2f}%')


#####

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 데이터셋 준비
# 손글씨 데이터셋 로드
digits = load_digits()
X = digits.data  # (n_samples, n_features)
y = digits.target  # (n_samples,)

# 텐서로 변환
X = torch.FloatTensor(X)
y = torch.LongTensor(y)

# 훈련 세트와 테스트 세트로 분할 (훈련 데이터 80%, 테스트 데이터 20%)
# 데이터를 학습용과 테스트용으로 나눕니다.
# 참고: Scikit-learn - train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
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
# 옵티마이저 설정
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    # 순전파
    outputs = model(X_train)
    # 손실 계산
    loss = criterion(outputs, y_train)
    # 역전파
    loss.backward()
    # 가중치 업데이트
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 4. 모델 평가
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = accuracy_score(y_test, predicted)
    print(f'\nTest Accuracy: {accuracy*100:.2f}%')


#####

import torch
import pandas as pd
from torch.utils.data import Dataset


# Dataset 클래스는 PyTorch에서 데이터셋을 정의할 때 사용하는 기본 클래스입니다.
# __len__() 메서드를 구현하여 데이터셋의 길이를 반환하고,
# __getitem__() 메서드를 구현하여 특정 인덱스의 데이터를 반환하는 방식으로 데이터셋을 정의할 수 있습니다.
# PyTorch의 DataLoader와 함께 사용하여 데이터를 쉽게 배치 단위로 처리할 수 있습니다.
# 참고 페이지: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
# Dataset을 상속받아 교통데이터에 맞게 사용할 수 있도록 수정합니다.
class TrafficDataset(Dataset):
    def __init__(self, excel_file, sheet_name=0):
        # 데이터를 로드 (엑셀 파일을 읽어와 pandas DataFrame으로 저장)
        self.data = pd.read_excel(excel_file, sheet_name=sheet_name)

        # 범주형 데이터를 숫자형으로 변환 (혼잡 여부: 0 또는 1로 변환)
        self.data['혼잡'] = self.data['혼잡'].astype(int)

        # 필요한 열만 선택하여 features에 저장 (7시,8시, 9시, 10시 시간대의 교통량)
        self.features = self.data[['7시', '8시', '9시', '10시']]

        # 혼잡 여부를 라벨로 설정 (이 데이터셋에서 타겟 값으로 사용)
        self.labels = self.data['혼잡']

    def __len__(self):
        # 데이터셋의 길이 반환 (전체 샘플 개수)
        return len(self.data)

    def __getitem__(self, idx):
        # 1. 특징(feature) 데이터를 가져옴
        sample_data = self.features.iloc[idx].values  # 특징 데이터를 가져옴 (NumPy 배열)

        # 2. 특징 데이터를 float 형으로 변환
        sample_data = sample_data.astype(float)

        # 3. 특징 데이터를 PyTorch 텐서로 변환
        # 참고: https://pytorch.org/docs/stable/generated/torch.tensor.html
        sample_tensor = torch.tensor(sample_data, dtype=torch.float32)

        # 4. 라벨(label)을 가져옴
        label_data = self.labels.iloc[idx]

        # 5. 라벨 데이터를 PyTorch 텐서로 변환
        # 참고: https://pytorch.org/docs/stable/generated/torch.tensor.html
        label_tensor = torch.tensor(label_data, dtype=torch.long)

        # 6. 텐서 형태의 특징과 라벨을 반환
        return sample_tensor, label_tensor


if __name__ == '__main__':
    # 데이터셋 예시 사용
    traffic_dataset = TrafficDataset('../data/weekday_traffic.xlsx')
    print(f"데이터셋 개수: {len(traffic_dataset)}")

    # 데이터셋에서 특정 인덱스의 데이터를 가져오기
    sample_data, sample_label = traffic_dataset[0]  # 첫 번째 데이터
    print(f"첫 번째 데이터 (특징): {sample_data}")
    print(f"첫 번째 데이터 (레이블): {sample_label}")

    sample_data, sample_label = traffic_dataset[5]  # 두 번째 데이터
    print(f"두 번째 데이터 (특징): {sample_data}")
    print(f"두 번째 데이터 (레이블): {sample_label}")
