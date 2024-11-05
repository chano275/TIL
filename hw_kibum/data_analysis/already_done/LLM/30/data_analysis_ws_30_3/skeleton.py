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
    random_state=42,  # 재현성을 위한 랜덤 시드를 설정합니다.
    use_rslora=False,  # Rank Stabilization LoRA를 활성화하려면 True로 설정합니다.
    loftq_config=None,  # LoftQ 설정을 지정합니다 (사용하는 경우).
)


####################################### 데이터 설정 #######################################


### 데이터 준비

# 데이터셋을 준비하는 것은 훈련에 매우 중요합니다. 모델이 무한히 생성하지 않도록 EOS_TOKEN(문장 종료)을 추가하는 것을 잊지 마세요.
# 그렇지 않으면 무한 생성이 발생할 수 있습니다.

# 완성된 텍스트만을 학습하고자 한다면, TRL의 문서참고
# https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only

"""
# load_dataset 함수를 사용하여 특정 데이터셋을 로드하고, 이를 특정 형식으로 포매팅하는 과정을 설명합니다.
# load_dataset 함수로 train 분할로 로드합니다.
# 데이터셋의 각 예제에 대해 formatting_prompts_func 함수를 적용하여 포매팅을 수행합니다.
# 이 함수는 instruction과 output 필드를 사용하여 주어진 포맷에 맞게 텍스트를 재구성합니다.
# 재구성된 텍스트는 prompt 포맷을 따르며, 각 항목의 끝에는 EOS_TOKEN을 추가하여 생성이 종료되도록 합니다.
# 최종적으로, 포매팅된 텍스트는 text 키를 가진 딕셔너리 형태로 반환됩니다.
# 이 과정을 통해, AI 모델이 처리하기 적합한 형태로 데이터를 전처리하는 방법을 보여줍니다.
"""

from datasets import load_dataset

# EOS_TOKEN은 문장의 끝을 나타내는 토큰입니다. 이 토큰을 추가해야 합니다.
EOS_TOKEN = tokenizer.eos_token

# Prompt를 사용하여 지시사항을 포맷팅하는 함수입니다.
prompt = """아래는 작업을 설명하는 지시 사항입니다. 요청에 적절하게 응답을 완성하세요.

### 지시 사항:
{}

### 응답:
{}
"""


# 주어진 예시들을 포맷팅하는 함수입니다.
# TODO: 포맷팅을 위해 옆의 설명에 맞게 빈칸을 채워주세요.
def formatting_prompts_func(examples):
    # 데이터셋에서 'instruction' 필드를 가져옵니다.
    instructions = ______________  # 지시사항을 가져옵니다.
    # 데이터셋에서 'output' 필드를 가져옵니다.
    outputs = ______________           # 출력값을 가져옵니다.
    texts = []                              # 포맷팅된 텍스트를 저장할 리스트입니다.
    for instruction, output in zip(instructions, outputs):
        # EOS_TOKEN을 추가해야 합니다. 추가하지 않으면 생성이 무한히 진행될 수 있습니다.
        text = prompt.format(instruction, output) + ______________
        texts.append(text)
    return {
        # 'text' 키를 가진 딕셔너리 형태로 반환합니다.
        "text": texts,  # 포맷팅된 텍스트를 반환합니다.
    }


# 데이터셋 파일 경로
jsonl_file = "traffic_qa_pair.jsonl"

# JSONL 파일에서 데이터셋을 로드합니다.
dataset = load_dataset("json", data_files=jsonl_file, split="train")

# 데이터셋에 포맷팅 함수를 적용합니다.
# TODO: datset을 만들기 위해 안에 들어갈 함수를 빈칸에 넣어주세요.
dataset = dataset.map(
    ______________, # 각 데이터에 대해서 formatting_prompts_func 함수를 적용합니다.
    batched=True,
)


# 데이터 포맷이 올바른지 확인하기 위해 샘플 데이터를 출력합니다.
print(dataset[0])                    # 첫 번째 예시를 출력합니다.
print(dataset.column_names)          # 데이터셋의 필드 이름을 확인합니다.
print(dataset["text"][:5])           # 변환된 'text' 필드의 내용을 확인합니다.
