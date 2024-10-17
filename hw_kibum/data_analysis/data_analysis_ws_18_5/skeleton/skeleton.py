import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

# 필요한 모듈 불러오기
# 해당 시간에는 목적함수를 다루는 것에 중점을 두기 위해 모델 학습을 위해 필요한 이전 실습에서 구현한 코드와 train 함수를 만들어놨습니다.
from traffic_model import TrafficDataset, TrafficMLP, set_seed

# 학습률 스케줄러를 사용하는 모델 학습 함수
def train_model_with_scheduler(device, model, dataloader, criterion, optimizer, scheduler, num_epochs=10):
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

        _________________________  # 학습률 스케줄러 업데이트 (참고: https://pytorch.org/docs/stable/optim.html#optimizer-step)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")
# 시드 고정
set_seed(42)

# 데이터 로드 및 DataLoader 설정
traffic_dataset = TrafficDataset('../data/weekday_traffic.xlsx')
traffic_loader = DataLoader(traffic_dataset, batch_size=8, shuffle=True)

# 모델 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU 또는 CPU 설정
model = TrafficMLP(input_size=3, hidden_size=64, output_size=2).to(device)  # 모델 인스턴스 생성

# 손실 함수 설정
"""
CrossEntropyLoss는 분류 문제에서 자주 사용되는 손실 함수로, 두 확률 분포 간의 차이를 계산합니다.
주로 이진 또는 다중 클래스 분류 문제에서 사용됩니다.
Softmax 활성화 함수를 내부적으로 포함하고 있어, 예측 값에 대해 확률 분포를 생성한 후, 실제 정답과의 차이를 계산합니다. 
클래스 간의 예측 확률과 실제 레이블 사이의 로그 손실을 측정하며, 예측이 실제 레이블에 가까울수록 손실 값은 낮아집니다.
CrossEntropyLoss는 신경망의 출력이 정규화된 확률 분포가 아닐 때, 사용하기 적합합니다.
"""
criterion = _________________________  # 크로스엔트로피 손실 함수
# 참고: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

# Adam 옵티마이저 설정
# 참고: https://pytorch.org/docs/stable/optim.html#module-torch.optim.adam
# learning rate는 0.001로 선언합니다.
optimizer = _________________________

# OneCycleLR 학습률 스케줄러 설정
"""
OneCycleLR은 학습률을 한번 크게 상승시키고 다시 줄이는 방식으로 학습의 효율성을 높입니다.

OneCycleLR 스케줄러는 다음과 같은 단계로 작동합니다:
Warm-up 단계: 처음 몇 에포크 동안 학습률을 점진적으로 증가시켜 모델이 학습 초기 과도하게 큰 학습률로부터 영향을 받지 않도록 합니다.
최대 학습률 도달: 학습률이 미리 설정한 max_lr 값에 도달한 후, 점차적으로 학습률을 감소시키기 시작합니다.
최종 감소: 학습 후반부에서는 학습률을 다시 크게 낮추어 학습이 안정적으로 수렴되도록 합니다.
"""
# OneCycleLR은 학습률을 한번 크게 상승시키고 다시 줄이는 방식으로 학습의 효율성을 높입니다.
# 스케쥴러의 max_lr은 0.01, step_per_epoch은 loader의 길이만큼, epoch은 학습과 같이 10으로 지정합니다.
scheduler = _________________________
# 참고: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.OneCycleLR

# 모델 학습 실행
train_model_with_scheduler(device, model, traffic_loader, criterion, optimizer, scheduler, num_epochs=10)