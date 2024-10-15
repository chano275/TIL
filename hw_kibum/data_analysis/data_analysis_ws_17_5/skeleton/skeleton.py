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

class TrafficDataset(Dataset):  ### 가지고 있는 데이터를 idx로 접근할 수 있게 해주는 class
    def __init__(self, excel_file, sheet_name=0):
        self.data = pd.read_excel(excel_file, sheet_name=sheet_name)          # 데이터를 로드 (엑셀 파일을 읽어와 pandas DataFrame으로 저장)
        self.data['혼잡'] = self.data['혼잡'].astype(int)          # 범주형 데이터를 숫자형으로 변환 (혼잡 여부: 0 또는 1로 변환)
        self.features = self.data[['8시', '9시', '10시']]          # 필요한 열만 선택하여 features에 저장 (8시, 9시, 10시 시간대의 교통량)
        self.labels = self.data['혼잡']          # 혼잡 여부를 라벨로 설정 (이 데이터셋에서 타겟 값으로 사용)

    def __len__(self):  return len(self.data)        # 데이터셋의 길이 반환 (전체 샘플 개수)

    ### 데이터 로더 : 학습을 시킬 때에는 SGD로 시킴 & 아래 설명에서 RANDOM하게 데이터 뽑아주는 애
    # GD = Gradient Descent : 역전파로 계산된 loss를 가지고 loss가 줄어드는 방향으로 param을 업데이트 하는 것
    # GD를 할 때에 DATA를 batch size(4)로 쪼개서 해당 4개 를 보고 loss를 계산 => 그 loss에 따라 update
    # 이 4개로 쪼개는걸 Stocastic 하게 한다 => SGD

    ### 모델 : 뉴럴넷

    def __getitem__(self, idx):
        # 1. 특징(feature) 데이터를 가져옴
        sample_data = self.features.iloc[idx].values      # 특징 데이터를 가져옴 (NumPy 배열)
        sample_data = sample_data.astype(float)          # 2. 특징 데이터를 float 형으로 변환
        sample_tensor = torch.tensor(sample_data, dtype=torch.float)        # 3. 특징 데이터를 PyTorch 텐서로 변환 & 참고: https://pytorch.org/docs/stable/generated/torch.tensor.html
        label_data = self.labels.iloc[idx]               # 4. 라벨(label)을 가져옴
        label_tensor = torch.tensor(label_data, dtype=torch.long)          # 5. 라벨 데이터를 PyTorch 텐서로 변환 & 참고: https://pytorch.org/docs/stable/generated/torch.tensor.html
        return sample_tensor, label_tensor               # 6. 텐서 형태의 특징과 라벨을 반환

### 모델 생성
# nn.Module을 상속받아 MLP 모델을 정의 (Multi-Layer Perceptron)
# nn.Module은 모든 신경망의 기본 클래스입니다. 신경망 레이어와 학습 가능한 파라미터를 관리하고, 순전파 연산을 정의하는 데 사용됩니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
class TrafficMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):  # nn.Module의 생성자를 호출하여 부모 클래스 초기화
        super(TrafficMLP, self).__init__()

        ### 3개의 layer 로 구성되어 있음
        self.fc1 = nn.Linear(input_size, hidden_size)   # 완전 연결층(fully connected layer) 정의 / input_size에서 hidden_size로 가는 선형 변환 (가중치 및 편향 포함)
        self.fc2 = nn.Linear(hidden_size, output_size)  # 두 번째 완전 연결층 정의: hidden_size에서 output_size로 가는 선형 변환

    def forward(self, x):      # 순전파 연산 정의: 입력을 받아 레이어를 통과시키는 연산을 정의합니다.
        x = F.relu(self.fc1(x))  # 첫 번째 완전 연결층(fc1)을 통과한 후 ReLU 활성화 함수 적용  /  ReLU는 비선형 활성화 함수로, 음수 값을 0으로, 양수는 그대로 반환합니다.# 참고: https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
        x = self.fc2(x)          # 두 번째 완전 연결층(fc2)을 통과한 후 출력 / 이 레이어에서는 출력값이 그대로 반환 / 보통 활성화 함수는 여기서는 사용되지 않음 (로짓값 출력)

        return x  # 최종 출력


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


# GPU/CPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU가 사용 가능하면 'cuda', 아니면 'cpu'로 모델 학습 장치 설정

# 모델 및 데이터셋 설정
model = TrafficMLP(input_size=3, hidden_size=64, output_size=2).to(device)    # TrafficMLP 모델 인스턴스를 생성하고 GPU/CPU로 전송
traffic_dataset = TrafficDataset('../data/weekday_traffic.xlsx')  # TrafficDataset을 사용하여 엑셀 데이터를 불러오고 DataLoader에 적용
traffic_loader = DataLoader(traffic_dataset, batch_size=32, shuffle=True)  # DataLoader는 데이터 배치를 관리하며, 학습 시 데이터를 섞어서 사용

criterion = nn.CrossEntropyLoss()  # 손실 함수 및 옵티마이저 설정 - CrossEntropyLoss는 분류 문제에서 자주 사용되는 손실 함수입니다. -  참고: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 옵티마이저는 가중치를 업데이트할 때 사용하는 방법 중 하나입니다. 참고: https://pytorch.org/docs/stable/optim.html#torch.optim.Adam

model.train()  # 학습 모드 설정 - model.train()은 모델을 학습 모드로 전환합니다. 참고: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train

for epoch in range(10):  # 총 10 에포크 동안 학습  # 간단한 학습 루프 (10 에포크를 기준으로 진행해봅니다.)
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
model.eval()  # 평가 모드 설정 model.eval()은 모델을 평가 모드로 전환합니다. 드롭아웃과 배치 정규화가 비활성화됩니다.  # 참고: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval
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
