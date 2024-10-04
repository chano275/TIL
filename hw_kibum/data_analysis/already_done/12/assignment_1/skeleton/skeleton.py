import torch

# requires_grad=True로 텐서 생성
# torch.randn() 함수를 사용하여 3차원 랜덤 텐서를 생성하고, requires_grad=True로 설정하여 그래디언트를 추적합니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.randn.html
x = ___________
print(f"텐서: {x}")

# 수식 설정 및 연산
# 텐서에 대한 연산을 설정합니다. 예를 들어, 텐서에 2를 곱하고 평균을 계산합니다.
y = ___________
z = ___________

# 역전파
# backward() 함수를 사용하여 z에 대한 그래디언트를 계산합니다.
_________

# 그래디언트 출력
# 계산된 텐서 x의 그래디언트를 출력합니다.
print(f"그래디언트: {x.grad}")