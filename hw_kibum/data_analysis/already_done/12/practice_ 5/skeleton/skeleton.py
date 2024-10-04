import numpy as np
import torch

# NumPy 배열을 텐서로 변환
# NumPy 배열을 텐서로 변환하기 위해 torch.from_numpy() 함수를 사용합니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.from_numpy.html
np_array = np.array([1, 2, 3])
tensor = ___________
print(f"NumPy에서 텐서로 변환: {tensor}")

# 텐서를 NumPy 배열로 변환
# 텐서를 다시 NumPy 배열로 변환하기 위해 텐서의 numpy() 메소드를 사용합니다.
# 참고: https://pytorch.org/docs/stable/tensors.html#torch.Tensor.numpy
np_array_from_tensor = ___________
print(f"텐서에서 NumPy로 변환: {np_array_from_tensor}")

# 메모리 공유 예시
# 텐서와 NumPy 배열은 메모리를 공유하기 때문에, 텐서의 값을 변경하면 NumPy 배열에도 변경이 반영됩니다.
tensor[0] = ___________
print(f"변경 후 NumPy 배열: {np_array_from_tensor}")  # NumPy 배열도 함께 변경됨
