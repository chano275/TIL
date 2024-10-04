import torch

# 임의의 텐서 생성
# 4x4 크기의 임의의 텐서를 생성하기 위해 torch.randn() 함수를 사용합니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.randn.html
tensor = ___________
print(f"텐서 크기: {tensor.shape}")

# 인덱싱으로 첫 번째 요소 추출
# 텐서의 첫 번째 요소인 [0, 0] 위치의 값을 추출하기 위해 인덱싱을 사용합니다.
index_tenssor = ___________
print(f"첫 번째 요소: {index_tenssor}")
print(f"첫 번째 요소 크기: {index_tenssor.shape}")

# 슬라이싱으로 두 번째, 세 번째 열의 요소 전부 추출한 새로운 텐서 생성
# 텐서에서 두 번째 열과 세 번째 열의 요소를 추출하기 위해 슬라이싱을 사용합니다.
sliced_tensor = ___________
print(f"슬라이싱 결과 (두 번째와 세 번째 열): {sliced_tensor}")
print(f"슬라이싱 결과 크기: {sliced_tensor.shape}")

# 슬라이싱한 부분 수정
# 슬라이싱된 텐서의 첫 번째 행 첫 번째 요소를 10으로 수정합니다.
# 그 결과, 원본 텐서가 어떻게 변했는지 확인합니다.
_______________________
print(f"수정 후 텐서: {tensor}")
