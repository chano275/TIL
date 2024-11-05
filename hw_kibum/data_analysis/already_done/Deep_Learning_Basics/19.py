import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ------------------------- 랜덤 시드 및 장치 설정 --------------------------------------------
# 재현성을 위해 랜덤 시드 고정
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# GPU 또는 CPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

# ------------------------- 데이터셋 로드 --------------------------------------------
# 로컬에 저장된 pickle 파일에서 데이터셋 불러오기
save_path = '../data/gtsrb_train_dataset_100_random.pkl'
with open(save_path, 'rb') as f:
    train_data = pickle.load(f)

print(f"로컬에서 불러온 데이터셋 샘플 수: {len(train_data)}")

# ------------------------- 전처리 및 데이터셋 클래스 정의 --------------------------------------------
# 이미지 전처리를 위한 변환 정의
transform_basic = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 그레이스케일 변환 정의
transform_grayscale = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 데이터 증강(Augmentation) 변환 정의
transform_augmentation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 커스텀 Dataset 클래스 정의
class GTSRBDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data  # 데이터 로드
        self.transform = transform  # 전처리 설정

    def __len__(self):
        return len(self.data)  # 데이터셋의 크기 반환

    def __getitem__(self, idx):
        image = self.data['image'][idx]  # 이미지 가져오기
        label = self.data['label'][idx]  # 라벨 가져오기
        if self.transform:
            image = self.transform(image)  # 전처리 적용
        return image, label  # 이미지와 라벨 반환

# ------------------------- 데이터 시각화 함수 정의 --------------------------------------------
def imshow(img, title="Image"):
    img = img / 2 + 0.5  # 정규화 해제
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray' if img.shape[0] == 1 else None)
    plt.title(title)
    plt.show()

def show_preprocessing_comparison(data, idx):
    # 원본 이미지
    original_image = data['image'][idx]
    plt.imshow(original_image)
    plt.title('전처리 전')
    plt.show()

    # 기본 전처리 후 이미지
    transformed_image = transform_basic(original_image)
    imshow(transformed_image, title='기본 전처리 후')

    # 그레이스케일 전처리 후 이미지
    transformed_image_grayscale = transform_grayscale(original_image)
    imshow(transformed_image_grayscale, title='그레이스케일 전처리 후')

def show_augmentation_comparison(data, idx):
    # 원본 이미지
    original_image = data['image'][idx]
    plt.imshow(original_image)
    plt.title('증강 전')
    plt.show()

    # 데이터 증강 후 이미지
    augmented_image = transform_augmentation(original_image)
    imshow(augmented_image, title='데이터 증강 후')

# ------------------------- DataLoader 설정 --------------------------------------------
# 데이터셋 인스턴스 생성
train_dataset = GTSRBDataset(train_data, transform=transform_basic)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ------------------------- 모델 정의 --------------------------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 합성곱 및 풀링 레이어 정의
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)  # 입력 채널 3개(RGB), 출력 채널 16개
        self.pool = nn.MaxPool2d(2, 2)          # 맥스풀링 레이어
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1) # 입력 채널 16개, 출력 채널 32개

        # 전결합층 정의
        self.fc1 = nn.Linear(32 * 56 * 56, 128) # 입력 노드 수는 합성곱 출력 크기에 따라 결정
        self.fc2 = nn.Linear(128, 43)           # 출력 노드 수는 클래스 수(43개)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 합성곱 -> ReLU -> 풀링
        x = self.pool(torch.relu(self.conv2(x)))  # 합성곱 -> ReLU -> 풀링
        x = x.view(-1, 32 * 56 * 56)              # 특징 맵을 일렬로 펼침
        x = torch.relu(self.fc1(x))               # 전결합층 -> ReLU
        x = self.fc2(x)                           # 출력층
        return x

# ------------------------- 모델 학습 준비 --------------------------------------------
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------- 데이터 확인 (옵션) --------------------------------------------
def check_data_loader(loader):
    data_iter = iter(loader)
    images, labels = next(data_iter)
    print(f"이미지 배치 크기: {images.shape}")
    print(f"라벨 배치: {labels}")

# DataLoader 확인
check_data_loader(train_loader)

# ------------------------- 전처리 및 증강 결과 시각화 --------------------------------------------
# 첫 번째 샘플에 대해 전처리 및 증강 결과 비교
show_preprocessing_comparison(train_data, idx=0)
show_augmentation_comparison(train_data, idx=0)

# ------------------------- 모델 학습 --------------------------------------------
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()  # 학습 모드 설정
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()       # 기울기 초기화
        outputs = model(images)     # 순전파
        loss = criterion(outputs, labels)  # 손실 계산
        loss.backward()             # 역전파
        optimizer.step()            # 가중치 업데이트

        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}')

# ------------------------- 학습 완료 후 저장 (옵션) --------------------------------------------
# 모델 저장
# torch.save(model.state_dict(), 'cnn_model.pth')
