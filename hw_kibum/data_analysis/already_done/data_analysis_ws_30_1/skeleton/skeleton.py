import torch

# CUDA 장치 기능 확인

# torch를 임포트하고 현재 사용 중인 CUDA 장치의 기능 버전을 확인합니다.
# 이 정보는 특정 라이브러리와의 호환성을 확인하고 최적의 성능을 보장하는 데 유용합니다.
# 현재 CUDA 장치의 major 및 minor 버전을 가져옵니다.
major_version, minor_version = torch.cuda.get_device_capability()
major_version, minor_version

### unsloth 라이브러리와 관련 디펜던시를 설치하는 과정을 설명합니다.
# Unsloth 및 종속성 설치하기
# 패키지 충돌을 방지하기 위해 별도로 설치해야 합니다.
"""
# TODO: 
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
major_version >= 8:
- 새로운 GPU(예: Ampere, Hopper GPUs - RTX 30xx, RTX 40xx, A100, H100, L40)에 사용
pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes

- 오래된 GPU(예: V100, Tesla T4, RTX 20xx)에 사용
pip install --no-deps xformers trl peft accelerate bitsandbytes
"""

# Unsloth: LLM의 파인튜닝을 최적화하기 위한 툴
"""
Unsloth는 Llama, Mistral, CodeLlama, TinyLlama, Vicuna, Open Hermes 등의 모델과 그 파생 모델을 지원합니다. 
16비트 LoRA 또는 4비트 QLoRA를 지원하며, 둘 다 전통적인 방법보다 2배 빠릅니다. 
max_seq_length는 내부적으로 RoPE 스케일링을 자동 처리하므로 어떤 값으로도 설정할 수 있습니다.
참고: https://docs.unsloth.ai/
"""


from unsloth import FastLanguageModel
import torch

# 최대 시퀀스 길이를 설정합니다; RoPE 스케일링은 내부적으로 자동 지원됩니다.
max_seq_length = 4096

# 자동 감지를 위해 None을 사용합니다. Tesla T4, V100은 Float16을, Ampere+는 Bfloat16을 사용하세요.
dtype = None

# 메모리 사용량을 줄이기 위해 4비트 양자화를 사용합니다; 필요 없다면 False로 설정할 수 있습니다.
load_in_4bit = True

# FastLanguageModel.from_pretrained 함수를 사용하여 사전 훈련된 언어 모델을 로드하는 과정
"""
- 최대 시퀀스 길이(max_seq_length)를 설정하여 모델이 처리할 수 있는 입력 데이터의 길이를 지정합니다.
- 데이터 타입(dtype)은 자동 감지되거나, 특정 하드웨어에 최적화된 형식(`Float16`, `Bfloat16`)으로 설정할 수 있습니다.
- 4비트 양자화(load_in_4bit) 옵션을 사용하여 메모리 사용량을 줄일 수 있으며, 이는 선택적입니다.
- 사전 정의된 4비트 양자화 모델 목록(fourbit_models)에서 선택하여 다운로드 시간을 단축하고 메모리 부족 문제를 방지할 수 있습니다.
- FastLanguageModel.from_pretrained 함수를 통해 모델과 토크나이저를 로드합니다. 
- 이때 모델 이름(`model_name`), 최대 시퀀스 길이, 데이터 타입, 4비트 로딩 여부를 매개변수로 전달합니다.
- 선택적으로, 특정 게이트 모델을 사용할 경우 토큰(`token`)을 제공할 수 있습니다.
"""
# 더 빠른 다운로드와 메모리 부족 방지를 위해 사전 양자화된 4비트 모델을 사용합니다.
# https://huggingface.co/unsloth/Llama-3.2-1B-bnb-4bit
# GPU를 고려하여 가장 가벼운 1b 모델을 선택합니다.
# 지원하는 모델 확인: https://docs.unsloth.ai/get-started/all-our-models 
# 모델과 토크나이저를 로드합니다.
# TODO: from_pretrained에 들어갈 인자를 채워주세요.
model, tokenizer = FastLanguageModel.from_pretrained(
    ______________="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",  # 모델 이름을 설정합니다. # Llama3.2 1b의 Instruct 버전
    ______________=max_seq_length,              # 최대 시퀀스 길이를 설정합니다.
    ______________=dtype,                                # 데이터 타입을 설정합니다.
    ______________=load_in_4bit,                  # 4비트 양자화 로드 여부를 설정합니다.
)
print(f"토크나이저: {tokenizer}")
print(f"모델: {model}")
