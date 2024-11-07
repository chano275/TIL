import torch
import numpy as np

# Pytorch로 구현된 AI모델 : 텐서 형식의 데이터를 입력받아야 처리 가능 > 리스트 [1,2,3]을 모델에 입력하기 위해 torch.tensor() 함수를 사용해 텐서로 변환
# 1D 텐서 생성 - 참고: https://pytorch.org/docs/stable/generated/torch.tensor.html
# 2D 텐서 생성 - torch.randn() 함수를 사용하여 (3, 4) 크기의 2D 텐서 생성- 참고: https://pytorch.org/docs/stable/generated/torch.randn.html
tensor1 = torch.tensor([1, 2, 3])  
tensor2 = torch.randn(3, 4)
# AI모델을 다루기 : 데이터의 크기(shape) / 자료형(dtype) 확인하는 것은 중요  -  모든 텐서가 같은 device에 존재해야 연산 가능 > 장치(device)도 확인
# 2D 텐서도 똑같이 데이터의 속성(크기, 자료형, 장치)을 확인
print(f"1D 텐서: {tensor1}, 크기: {tensor1.shape}, 자료형: {tensor1.dtype}, 장치: {tensor1.device}")
print(f"2D 텐서: {tensor2}, 크기: {tensor2.shape}, 자료형: {tensor2.dtype}, 장치: {tensor2.device}")
# 두 개의 랜덤 텐서 생성 - torch.rand() 함수를 사용하여 3x3 크기의 랜덤 텐서 생성 - 참고: https://pytorch.org/docs/stable/generated/torch.rand.html
a, b = torch.rand(3, 3), torch.rand(3, 3)


# 텐서 덧셈 - 두 텐서를 더한 결과를 저장 - 일반적으로 사용하는 연산 : out-of-place 방식으로 이뤄짐
# 텐서 뺄셈 - 두 텐서를 뺀 결과를 방식으로 저장 - 일반적으로 사용하는 연산 : out-of-place 방식으로 
# 텐서 곱셈 - 두 텐서를 곱한 결과를 방식으로 저장 - out-of-place 방식
c, d, e = a + b, a - b, a * b
print(f"덧셈 결과: {c} / 뺄셈 결과: {d} / 곱셈 결과: {e}")
# In-place 연산 예시 : add_() [ 결과가 기존 텐서 a에 바로 반영 ] > 메모리를 절약할 수 있지만, 기존 텐서의 값을 변경하므로 주의 - 참고: https://pytorch.org/docs/stable/generated/torch.Tensor.add_.html
a.add_(b)
print(f"In-place 덧셈 결과: {a}")


# 임의의 텐서 생성 - 4x4 크기의 임의의 텐서를 생성하기 위해 torch.randn() 함수 사용 - 참고: https://pytorch.org/docs/stable/generated/torch.randn.html
# 인덱싱으로 첫 번째 요소 추출 - 텐서의 첫 번째 요소인 [0, 0] 위치의 값을 추출하기 위해 인덱싱을 사용
# 슬라이싱으로 2, 3번째 열의 요소 전부 추출한 새로운 텐서 생성
# 슬라이싱한 부분 수정 - 슬라이싱된 텐서의 첫 번째 행 첫 번째 요소를 10으로 수정  >  원본 텐서가 어떻게 변했는지 확인
tensor = torch.randn(4, 4)
index_tenssor = tensor[0, 0]
sliced_tensor = tensor[:, 1:3]
sliced_tensor[0, 0] = 10
print(f"텐서 크기: {tensor.shape}")
print(f"첫 번째 요소: {index_tenssor}")
print(f"첫 번째 요소 크기: {index_tenssor.shape}")
print(f"슬라이싱 결과 (두 번째와 세 번째 열): {sliced_tensor}")
print(f"슬라이싱 결과 크기: {sliced_tensor.shape}")
print(f"수정 후 텐서: {tensor}")


# 임의의 텐서 생성[torch.randn()] - 2x3x4 크기의 임의의 텐서를 생성 - 참고: https://pytorch.org/docs/stable/generated/torch.randn.html
# 텐서 변형[reshape()] - 텐서를 4x6 크기로 변형
# 텐서 변형[view()] - 텐서를 1차원으로 변형 / 참고: https://pytorch.org/docs/stable/tensor_view.html#tensor-view-doc
# 텐서 변형[squeeze / unsqueeze] - 참고: https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
# unsqueeze() : 첫 번째 차원에 새로운 차원을 추가 
# squeeze()   : 크기가 1인 차원을 제거
tensor = torch.randn(2, 3, 4)
reshaped_tensor = tensor.reshape(6, 4)
viewed_tensor = tensor.view(-1)
unsqueezed_tensor = tensor.unsqueeze(0)
squeezed_tensor = unsqueezed_tensor.squeeze()
print(f'텐서: {tensor}, 텐서 크기: {tensor.shape}')
print(f"Reshape 결과: {reshaped_tensor}, Reshape 결과 크기: {reshaped_tensor.shape}")
print(f"View 결과: {viewed_tensor}, View 결과 크기: {viewed_tensor.shape}")
print(f"Unsqueeze 결과: {unsqueezed_tensor}, Unsqueeze 결과 크기: {unsqueezed_tensor.shape}")
print(f"Squeeze 결과: {squeezed_tensor}, Squeeze 결과 크기: {squeezed_tensor.shape}")


# NumPy 배열을 텐서로 변환 [ torch.from_numpy() ] - 참고: https://pytorch.org/docs/stable/generated/torch.from_numpy.html
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)
print(f"NumPy에서 텐서로 변환: {tensor}")

# 텐서를 NumPy 배열로 변환 - 텐서의 numpy() 메소드를 사용 - 참고: https://pytorch.org/docs/stable/tensors.html#torch.Tensor.numpy
np_array_from_tensor = tensor.numpy()
print(f"텐서에서 NumPy로 변환: {np_array_from_tensor}")

# 텐서와 NumPy 배열은 메모리를 공유 > 텐서의 값을 변경하면 NumPy 배열에도 변경이 반영
tensor[0] = 10
print(f"변경 후 NumPy 배열: {np_array_from_tensor}")  # NumPy 배열도 함께 변경됨


# torch.randn() 함수를 사용하여 3차원 랜덤 텐서 생성 > requires_grad=True로 설정하여 그래디언트를 추적
# 모델 업데이트를 위해서는 gradient를 계산해야 > gradient를 계산하기 위해서는 computatioonal graph가 필요 
# requires_grad=True로 설정 > torch연산을 수행할 때마다 해당 tensor에 대한 computational graph를 생성
# 참고: https://pytorch.org/docs/stable/generated/torch.randn.html
x = torch.randn(3, requires_grad=True)
print(f"텐서: {x}")

# 수식 설정 및 연산 - 텐서에 대한 연산을 설정합니다.  텐서에 2를 곱하고 평균을 계산 => y 에 x *2 를 연산하여 대입하고, z에 y의 평균 [ mean 함수 ] 을 대입 - 참고: https://pytorch.org/docs/stable/tensors.html#torch.Tensor.mean
y = x * 2
z = y.mean()

# 역전파 - backward() 함수를 사용 > z에 대한 그래디언트를 계산 - 참고: https://pytorch.org/docs/stable/generated/torch.mean.html#torch.mean
z.backward()

# 계산된 텐서 x의 그래디언트를 출력
print(f"그래디언트: {x.grad}")


# CPU에서 텐서 생성 - 참고: https://pytorch.org/docs/stable/generated/torch.randn.html
cpu_tensor = torch.randn(3, 3)
print(f"CPU 텐서: {cpu_tensor}, 장치: {cpu_tensor.device}")

# GPU로 텐서 이동 [ torch.cuda.is_available() ]  >  GPU가 사용 가능한지 확인 > 사용 가능하다면 텐서를 'cuda'로 이동시킴 - 참고: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.to('cuda')
    print(f"GPU 텐서: {gpu_tensor}, 장치: {gpu_tensor.device}")
else:
    print("GPU가 지원되지 않습니다.")
