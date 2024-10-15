import torch
import torch.nn as nn
import torch.nn.functional as F


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