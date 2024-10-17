import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# 필요한 모듈 불러오기
# 해당 시간에는 목적함수를 다루는 것에 중점을 두기 위해 모델 학습을 위해 필요한 이전 실습에서 구현한 코드와 train 함수를 만들어놨습니다.
from traffic_model import TrafficDataset, TrafficMLP, train_model, set_seed

# 시드 고정
set_seed(42)  # 재현성을 위해 랜덤 시드 고정

# 데이터 로드 및 DataLoader 설정
# DataLoader는 대용량 데이터를 효과적으로 배치(batch)로 불러와서 모델 학습에 사용할 수 있도록 합니다.
traffic_dataset = TrafficDataset('../data/weekday_traffic.xlsx')
traffic_loader = DataLoader(traffic_dataset, batch_size=8, shuffle=True)

# 모델 설정 (교통 데이터를 3개의 입력 특징으로 사용하여 이진 분류)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU 사용 여부 확인
model = TrafficMLP(input_size=3, hidden_size=64, output_size=2).to(device)  # MLP 모델 인스턴스 생성

# 손실 함수 설정
"""
CrossEntropyLoss는 분류 문제에서 자주 사용되는 손실 함수로, 두 확률 분포 간의 차이를 계산합니다.
주로 이진 또는 다중 클래스 분류 문제에서 사용됩니다.
Softmax 활성화 함수를 내부적으로 포함하고 있어, 예측 값에 대해 확률 분포를 생성한 후, 실제 정답과의 차이를 계산합니다. 
클래스 간의 예측 확률과 실제 레이블 사이의 로그 손실을 측정하며, 예측이 실제 레이블에 가까울수록 손실 값은 낮아집니다.
CrossEntropyLoss는 신경망의 출력이 정규화된 확률 분포가 아닐 때, 사용하기 적합합니다.
"""
criterion = _________________________   # 크로스엔트로피 손실 함수
# 참고: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

# Adam 옵티마이저 설정
# Adam 옵티마이저는 모멘텀과 학습률 조정을 자동으로 수행하는 알고리즘입니다.
# learning rate는 0.001로 선언합니다.
optimizer = _________________________
# 참고: https://pytorch.org/docs/stable/optim.html#torch.optim.Adam

# 모델 학습 실행
train_model(device, model, traffic_loader, criterion, optimizer, num_epochs=10)
# 학습 함수는 주어진 데이터 로더와 손실 함수, 옵티마이저를 사용하여 모델을 10번의 에포크 동안 학습합니다.
