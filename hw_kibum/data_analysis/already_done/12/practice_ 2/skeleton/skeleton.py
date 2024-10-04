import torch

# 두 개의 랜덤 텐서 생성
# torch.rand() 함수를 사용하여 3x3 크기의 랜덤 텐서를 생성합니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.rand.html
a = ___________
b = ___________

# 텐서 덧셈
# 두 텐서를 더한 결과를 저장합니다.
# 우리가 일반적으로 사용하는 연산은 out-of-place 방식으로 이루어집니다.
c = ___________
print(f"덧셈 결과: {c}")

# 텐서 뺄셈
# 두 텐서를 뺀 결과를 방식으로 저장합니다.
# 우리가 일반적으로 사용하는 연산은 out-of-place 방식으로 이루어집니다.
d = ___________
print(f"뺄셈 결과: {d}")

# 텐서 곱셈
# 두 텐서를 곱한 결과를 방식으로 저장합니다.
# 우리가 일반적으로 사용하는 연산은 out-of-place 방식으로 이루어집니다.
e = ___________
print(f"곱셈 결과: {e}")

# In-place 연산 예시
# add_()는 In-place 연산으로, 결과가 기존 텐서 a에 바로 반영됩니다.
# 이러한 연산은 메모리를 절약할 수 있지만, 기존 텐서의 값을 변경하므로 주의해야 합니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.Tensor.add_.html
a.___________
print(f"In-place 덧셈 결과: {a}")
