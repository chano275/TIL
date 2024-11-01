import torch

# 예시 데이터 생성 - y = 2 * x + 3 의 선형 회귀 모델을 학습시키기 위한 데이터
# x_data는 독립 변수, y_data는 종속 변수로서 모델이 학습할 실제 데이터입니다.
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # 입력 데이터 (x 값)
y_data = torch.tensor([[5.0], [7.0], [9.0], [11.0]])  # 출력 데이터 (y 값)

# 모델 초기화: 선형 모델 y = wx + b - w와 b는 학습을 통해 모델이 찾으려는 가중치와 바이어스입니다.
# requires_grad=True로 설정하면 PyTorch가 자동으로 기울기를 계산해 줍니다.
w = torch.tensor([0.0], requires_grad=True)  # 가중치 (w)
b = torch.tensor([0.0], requires_grad=True)  # 바이어스 (b)

# 학습률 설정 -  학습률은 파라미터를 얼마나 크게 업데이트할지를 결정합니다.
learning_rate = 0.01

# SGD 구현
for epoch in range(100):  # 총 100번의 epoch 동안 모델을 학습시킵니다.
    y_pred = x_data * w + b      # 순전파 계산 (예측값 계산) - 모델의 예측값은 y_pred = wx + b 로 계산됩니다.
    loss = ((y_pred - y_data) ** 2).mean()  # 손실 값 계산      # 손실 함수 (MSE) - 예측값과 실제 값 사이의 차이를 평균 제곱 오차(MSE)를 통해 계산합니다.
    loss.backward()      # 역전파: 기울기 계산 - 역전파 단계에서 손실에 대한 w와 b의 기울기를 계산합니다. 자동으로 w.grad와 b.grad에 기울기가 저장됩니다.

    # 파라미터 업데이트 (SGD 적용) - 손실 함수의 기울기를 사용해 파라미터(w, b)를 업데이트합니다.
    # torch.no_grad()는 기울기 추적을 중지시키는 컨텍스트로, 파라미터 업데이트 시 필요합니다.
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # 기울기 초기화
    # 기울기를 초기화하지 않으면 이전 epoch에서 계산된 기울기가 계속 누적되므로
    # 다음 epoch 학습에 영향을 주게 됩니다. 이를 방지하기 위해 초기화합니다.
    w.grad.zero_()
    b.grad.zero_()

    # 학습 중간 결과 출력
    # 10번째 epoch마다 w, b 값과 손실 값을 출력하여 학습 상황을 확인합니다.
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}: w = {w.item():.3f}, b = {b.item():.3f}, Loss = {loss.item():.4f}')

# 최종 모델 파라미터 출력
# 최종적으로 학습된 w와 b 값을 출력하여 학습이 잘 되었는지 확인합니다.
print(f'최종 모델: w = {w.item():.3f}, b = {b.item():.3f}')
