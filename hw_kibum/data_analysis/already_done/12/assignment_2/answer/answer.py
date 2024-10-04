import torch

# CPU에서 텐서 생성
# torch.randn() 함수를 사용하여 3x3 크기의 텐서를 생성하고, CPU에 위치한 텐서임을 확인합니다.
# 참고: https://pytorch.org/docs/stable/generated/torch.randn.html
cpu_tensor = torch.randn(3, 3)
print(f"CPU 텐서: {cpu_tensor}, 장치: {cpu_tensor.device}")

# GPU로 텐서 이동
# torch.cuda.is_available()를 사용하여 GPU가 사용 가능한지 확인한 후, GPU가 사용 가능하다면 텐서를 'cuda'로 이동시킵니다.
# 참고: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.to('cuda')
    print(f"GPU 텐서: {gpu_tensor}, 장치: {gpu_tensor.device}")
else:
    print("GPU가 지원되지 않습니다.")
