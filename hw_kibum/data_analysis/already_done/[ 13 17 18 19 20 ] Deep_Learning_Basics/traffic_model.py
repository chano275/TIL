import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import random
import numpy as np


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


# 학습 함수 정의
def train_model(device, model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")


# 시드 설정 함수
def set_seed(seed=42):
    """랜덤 시드를 고정하여 학습의 일관성과 재현성을 보장합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
