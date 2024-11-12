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


####################################### LoRA 설정 #######################################
# LoRA
# LoRA (Low-Rank Adaptation) 어댑터를 사용하면 모델 파라미터의 1%에서 10%만 업데이트하여 효율적으로 미세 조정할 수 있습니다.

# FastLanguageModel을 사용하여 특정 모듈에 대한 성능 향상 기법을 적용한 모델을 구성합니다.

"""
- FastLanguageModel.get_peft_model 함수를 호출하여 모델을 초기화하고, 성능 향상을 위한 여러 파라미터를 설정합니다.
- r 파라미터를 통해 성능 향상 기법의 강도를 조절합니다. 권장 값으로는 8, 16, 32, 64, 128 등이 있습니다.
- target_modules 리스트에는 성능 향상을 적용할 모델의 모듈 이름들이 포함됩니다.
- lora_alpha와 lora_dropout을 설정하여 LoRA(Low-Rank Adaptation) 기법의 세부 파라미터를 조정합니다.
- bias 옵션을 통해 모델의 바이어스 사용 여부를 설정할 수 있으며, 최적화를 위해 none으로 설정하는 것이 권장됩니다.
- use_gradient_checkpointing 옵션을 unsloth로 설정하여 VRAM 사용량을 줄이고, 더 큰 배치 크기로 학습할 수 있도록 합니다.
- use_rslora 옵션을 통해 Rank Stabilized LoRA를 사용할지 여부를 결정합니다.
"""
from unsloth import FastLanguageModel

# FastLanguageModel을 사용하여 성능 향상 기법을 적용한 모델을 구성합니다.
# 참고: https://docs.unsloth.ai/basics/continued-pretraining#continued-pretraining-and-finetuning-the-lm_head-and-embed_tokens-matrices
# 참고: https://docs.unsloth.ai/basics/lora-parameters-encyclopedia#q_proj-query-projection
# TODO: 빈칸에 맞게 위의 참고 자료를 활용해서 모델을 완성하세요.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA의 랭크를 조정합니다; 권장 값은 8, 16, 32, 64, 128입니다.
    lora_alpha=32,  # LoRA 알파 값을 설정합니다.
    lora_dropout=0.05,  # LoRA 레이어에 드롭아웃 비율을 지정합니다.
    target_modules = [
        "_______",   # 쿼리 프로젝션: 입력을 쿼리 공간으로 변환하여 주의(attention) 점수 계산에 사용
        "_______",   # 키 프로젝션: 입력을 키 공간으로 변환하여 쿼리 벡터와 비교해 주의 가중치 결정
        "_______",   # 값(value) 프로젝션: 입력을 값 공간으로 변환하여 주의 가중치로 결합된 출력을 생성
        "_______",   # 출력 프로젝션: 결합된 값을 입력 차원으로 변환하여 주의 결과를 모델에 통합
        "_______",   # 게이트 프로젝션: 게이트 메커니즘에서 정보 흐름 제어
        "_______",   # 업 프로젝션: 입력 차원을 확장, 주로 피드포워드 계층에서 사용
        "_______",   # 다운 프로젝션: 입력 차원을 축소하여 계산 복잡도 감소 및 모델 크기 제어
    ],
 
    bias="none",  # 바이어스 사용 여부를 지정합니다; 최적화를 위해 "none"을 권장합니다.
    use_gradient_checkpointing="unsloth",  # VRAM 사용량을 줄이고 더 큰 배치 크기를 지원합니다.
    random_state=123,  # 재현성을 위한 랜덤 시드를 설정합니다.
    use_rslora=False,  # Rank Stabilization LoRA를 활성화하려면 True로 설정합니다.
    loftq_config=None,  # LoftQ 설정을 지정합니다 (사용하는 경우).
)

print(f"LoRA 어댑터가 추가된 모델: {model}")