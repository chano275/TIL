import torch

# 임의의 텐서 생성
# 2x3x4 크기의 임의의 텐서를 생성하기 위해 torch.randn() 함수를 사용합니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.randn.html
tensor = torch.randn(2, 3, 4)
print(f'텐서: {tensor}')
print(f'텐서 크기: {tensor.shape}')

# 텐서 변형: reshape
# 텐서를 4x6 크기로 변형하기 위해 reshape() 함수를 사용합니다.
reshaped_tensor = tensor.reshape(6, 4)
print(f"Reshape 결과: {reshaped_tensor}")
print(f"Reshape 결과 크기: {reshaped_tensor.shape}")

# 텐서 변형: view
# 텐서를 1차원으로 변형하기 위해 view() 함수를 사용합니다.
# 참고: https://pytorch.org/docs/stable/tensor_view.html#tensor-view-doc
viewed_tensor = tensor.view(-1)
print(f"View 결과: {viewed_tensor}")
print(f"View 결과 크기: {viewed_tensor.shape}")

# 텐서 변형: squeeze 및 unsqueeze
# 첫 번째 차원에 새로운 차원을 추가하기 위해 unsqueeze() 함수를 사용합니다.
# 그 후, 크기가 1인 차원을 제거하기 위해 squeeze() 함수를 사용합니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
unsqueezed_tensor = tensor.unsqueeze(0)
print(f"Unsqueeze 결과: {unsqueezed_tensor}")
print(f"Unsqueeze 결과 크기: {unsqueezed_tensor.shape}")

# 차원이 1인 차원을 제거하기 위해 squeeze() 함수를 사용합니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.squeeze.html
squeezed_tensor = unsqueezed_tensor.squeeze()
print(f"Squeeze 결과: {squeezed_tensor}")
print(f"Squeeze 결과 크기: {squeezed_tensor.shape}")
