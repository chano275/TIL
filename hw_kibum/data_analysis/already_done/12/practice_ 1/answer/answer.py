# PyTorch 라이브러리를 임포트 하기 전에 설치한다. (pip install torch)
import torch

# 1D 텐서 생성
# Pytorch로 구현된 인공지능 모델은 텐서 형식의 데이터를 입력받아야 처리할 수 있습니다.
# 따라서 리스트 [1,2,3]을 모델에 입력하기 위해서 torch.tensor() 함수를 사용해서 텐서로 바꿔줍니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.tensor.html
tensor1 = torch.tensor([1, 2, 3])

# 인공지능 모델을 다루기 위해 데이터의 크기(shape)와 자료형(dtype)을 확인하는 것은 중요합니다.
# 또한 모든 텐서는 같은 device에 존재해야 연산이 가능하므로 장치(device)도 확인합니다.
print(f"1D 텐서: {tensor1}, 크기: {tensor1.shape}, 자료형: {tensor1.dtype}, 장치: {tensor1.device}")

# 2D 텐서 생성
# torch.randn() 함수를 사용하여 2D 텐서를 생성합니다. 크기는 (3, 4)입니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.randn.html
tensor2 = torch.randn(3, 4)

# 이번에 생성한 2D 텐서도 똑같이 데이터의 속성(크기, 자료형, 장치)을 확인합니다.
print(f"2D 텐서: {tensor2}, 크기: {tensor2.shape}, 자료형: {tensor2.dtype}, 장치: {tensor2.device}")
