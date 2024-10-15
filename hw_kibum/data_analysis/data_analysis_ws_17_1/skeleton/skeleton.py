# PyTorch 설치 (CUDA 지원이 필요한 경우)
# pip install torch

# PyTorch 모듈 임포트
import torch

# GPU 사용 가능 여부 확인
# 참고 페이지: https://pytorch.org/docs/stable/cuda.html#torch.cuda.is_available
if torch.cuda.is_available():
    # CUDA가 지원되는 GPU가 있는지 확인 (True/False 반환)
    # 참고 페이지: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
    device = _______________________  # GPU 장치를 사용할 수 있으면 'cuda'로 설정
    # 사용 가능한 첫 번째 GPU 장치의 이름 출력
    # 참고 페이지: https://pytorch.org/docs/stable/cuda.html#torch.cuda.get_device_name
    print(f"GPU 사용 가능: {_______________________}")
else:
    # CUDA가 없으면 'cpu'로 설정
    # 참고 페이지: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
    device = _______________________
    print("GPU 사용 불가, CPU 사용")
