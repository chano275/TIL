import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ------------------------- 커스텀 Dataset 클래스 정의 --------------------------------------------
# PyTorch의 Dataset 클래스를 상속받아 데이터셋을 정의합니다.
# 이 클래스는 '__len__' 메서드를 통해 데이터셋의 크기를 반환하고,
# '__getitem__' 메서드를 통해 특정 인덱스의 이미지와 라벨을 반환합니다.
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html 참고
class GTSRBDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data  # pickle로 불러온 데이터를 저장
        self.transform = transform  # 전처리 설정을 저장

    def __len__(self):
        return len(self.data)  # 데이터셋의 크기 반환

    def __getitem__(self, idx):
        # i번째 이미지 데이터를 self.data['image'] 리스트에서 가져옵니다.
        image = ___________________  # i번째 image를 저장된 self.data attribute에서 가져올 수 있습니다.

        # i번째 이미지의 라벨을 self.data['label'] 리스트에서 가져옵니다.
        label = ___________________  # i번째 label을 저장된 self.data attribute에서 가져올 수 있습니다.

        # 전처리가 설정된 경우, 이미지에 지정된 변환(transform)을 적용합니다.
        if self.transform:
            image = ___________________  # 이미지 전처리 수행
        return image, label  # 전처리된 이미지와 해당 이미지의 라벨을 반환합니다.



# ------------------------- 전처리 설정 --------------------------------------------
# torchvision.transforms 모듈의 Compose 함수를 사용하여 일련의 이미지 전처리 변환을 적용합니다.
# https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Compose 참고
# Compose는 여러 전처리 단계를 순차적으로 적용할 수 있도록 해주는 유용한 함수입니다.
# 예를 들어, 이미지를 텐서로 변환하고 정규화하는 과정을 한 번에 처리할 수 있습니다.

# V1 API 기준으로 파악
transform = transforms.Compose([
    # 참고 https://pytorch.org/vision/stable/transforms.html#v1-api-reference
    
    # transforms.Resize() 함수는 torchvision.transforms의 Resize 함수를 사용하여 
    # 이미지의 크기를 224x224로 조정합니다.
    transforms.___________________,

    # transforms.ToTensor() 함수는 torchvision.transforms의 ToTensor 함수를 사용하여 
    # 이미지를 PyTorch 텐서로 변환합니다.
    transforms.___________________,
    
    # transforms.Normalize(mean, std)는 이미지의 각 채널에 대해 지정된 평균(mean)과 표준편차(std)로 정규화합니다. 모두 0.5로 맞춰주세요.
    transforms.___________________  
    # 각 채널(R, G, B)의 평균을 0.5, 표준편차를 0.5로 설정하여 정규화. 정규화는 학습을 더 빠르고 안정적으로 진행하게 도와줍니다.

])

# ------------------------- CNN 모델 정의 --------------------------------------------
# CNN 모델 클래스 정의
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # 합성곱 층 1: 입력 채널 3(RGB 이미지), 출력 채널 16, 커널 크기 3x3, 패딩 1
        # 3개의 입력 채널(RGB)에서 16개의 출력 채널로 변환. 커널 크기는 3x3, 패딩 1은 이미지의 크기를 유지시켜줌.
        # 참고: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv1 = nn.Conv2d(in_channels=___, out_channels=___, kernel_size=___, stride=___, padding=___)

        # Max Pooling: 커널 크기 2x2, 스트라이드 2 (이미지 크기를 절반으로 줄임)
        # 풀링 층은 이미지의 크기를 줄이면서 중요한 특징을 추출합니다. 크기를 절반으로 줄임.
        # 참고: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        self.pool = nn.MaxPool2d(kernel_size=___, stride=___, padding=___)

        # 합성곱 층 2: 입력 채널 16, 출력 채널 32, 커널 크기 3x3, 패딩 1
        # 두 번째 합성곱 층은 16개의 입력 채널을 받아 32개의 출력 채널을 생성합니다. 커널 크기와 패딩은 동일.
        self.conv2 = nn.Conv2d(in_channels=___, out_channels=___, kernel_size=___, stride=___, padding=___)

        # Fully connected layer 1: 입력 크기 32 * 56 * 56, 출력 크기 128
        # Flatten 이후 이미지의 크기가 32개의 채널에 56x56이 됩니다. 이를 128개의 뉴런으로 연결.
        # 참고: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc1 = nn.Linear(32 * 56 * 56, ___)

        # Fully connected layer 2: 출력 크기 43 (GTSRB 데이터셋에 있는 43개의 교통 표지판 클래스)
        # 최종 출력은 43개의 교통 표지판 클래스에 대한 예측 값입니다.
        self.fc2 = nn.Linear(___, ___)

    def forward(self, x):
        # 합성곱 1 -> ReLU -> 풀링
        # ReLU 활성화 함수는 음수 값을 0으로 만듦으로써 비선형성을 추가합니다.
        # 참고: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        x = self.pool(torch.relu(self.conv1(x)))

        # 합성곱 2 -> ReLU -> 풀링
        x = self.pool(torch.relu(self.conv2(x)))

        # Flatten: 이미지를 1D 벡터로 변환 (FC층에 넣기 위해)
        # view(-1, ...)는 텐서를 1D 벡터로 변환합니다.
        # 참고: https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        x = x.view(-1, ___ * ___ * ___)

        # Fully connected layer 1 -> ReLU
        x = torch.relu(self.fc1(x))

        # Fully connected layer 2 (최종 출력, 클래스에 대한 확률 출력)
        x = self.fc2(x)

        return x



# ------------------------- 로컬 파일에서 데이터셋 불러오기 --------------------------------------------
# pickle 파일 경로를 지정하여 저장된 데이터셋을 로드합니다.
# pickle은 Python의 객체를 파일로 저장하고 다시 불러올 수 있는 직렬화/역직렬화 도구입니다.
save_path = '../data/gtsrb_train_dataset_100_random.pkl'  # pickle 파일 경로
with open(save_path, 'rb') as f:  # 파일을 'rb' 모드로 열어 바이너리 읽기 방식으로 불러옵니다.
    train_dataset = pickle.load(f)  # pickle로 데이터를 불러와 train_dataset에 저장합니다.


# ------------------------- DataLoader 설정 --------------------------------------------
# DataLoader는 배치 단위로 데이터를 로드하며, 학습 시 데이터를 효율적으로 불러올 수 있도록 합니다.
# batch_size=32는 한 번에 32개의 이미지를 로드하겠다는 의미이며, shuffle=True는 데이터의 순서를 무작위로 섞습니다.
train_dataset = GTSRBDataset(train_dataset, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 학습에 사용할 장치 설정 (GPU가 있으면 CUDA, 없으면 CPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)  # 모델을 설정한 장치로 이동

# 손실 함수 정의: 교차 엔트로피 손실
criterion = nn.CrossEntropyLoss()

# 옵티마이저 정의: Adam 옵티마이저, 학습률 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------- 학습 루프 --------------------------------------------

# 에포크 수 설정
num_epochs = 10

# 학습 과정
for epoch in range(num_epochs):
    running_loss = 0.0  # 에포크 동안의 손실을 기록
    # DataLoader에서 배치 단위로 데이터를 가져옴
    for images, labels in train_loader:
        # 이미지를 장치에 맞게 설정 (GPU 또는 CPU)
        images, labels = images.to(device), labels.to(device)
        # 경사도 초기화
        optimizer.zero_grad()
        # 순전파: 모델을 통해 예측값 계산
        outputs = model(images)
        # 손실 계산
        loss = criterion(outputs, labels)
        # 역전파: 손실에 따른 경사 계산
        loss.backward()
        # 옵티마이저를 사용하여 가중치 업데이트
        optimizer.step()
        # 손실 누적
        running_loss += loss.item()
    # 에포크당 평균 손실 출력
    avg_train_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}')