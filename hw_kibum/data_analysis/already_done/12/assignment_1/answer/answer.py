import torch

# requires_grad=True로 텐서 생성
# torch.randn() 함수를 사용하여 3차원 랜덤 텐서를 생성하고, requires_grad=True로 설정하여 그래디언트를 추적합니다.
# 모델 업데이트를 위해서는 gradient를 계산해야합니다. gradient를 계산하기 위해서는 computatioonal graph가 필요한데,
# requires_grad=True로 설정하면, torch연산을 수행할 때마다 해당 tensor에 대한 computational graph를 생성하게 됩니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.randn.html
x = torch.randn(3, requires_grad=True)
print(f"텐서: {x}")

# 수식 설정 및 연산
# 텐서에 대한 연산을 설정합니다.  텐서에 2를 곱하고 평균을 계산합니다.
# y 에 x *2 를 연산하여 대입하고, z에 y의 평균을 대입합니다.
# 텐서에서 평균은 mean() 함수를 사용하여 계산합니다.
# 참고: https://pytorch.org/docs/stable/tensors.html#torch.Tensor.mean
y = x * 2
z = y.mean()

# 역전파
# backward() 함수를 사용하여 z에 대한 그래디언트를 계산합니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.mean.html#torch.mean
z.backward()

# 그래디언트 출력
# 계산된 텐서 x의 그래디언트를 출력합니다.
print(f"그래디언트: {x.grad}")
