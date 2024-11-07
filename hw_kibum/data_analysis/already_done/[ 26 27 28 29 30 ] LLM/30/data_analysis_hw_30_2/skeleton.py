import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# 모델과 토크나이저를 로드하는 함수
def load_model_and_tokenizer(model_name):
    """
    사전 훈련된 LLM 모델과 토크나이저를 로드합니다.

    Parameters:
        model_name (str): 사전 훈련된 모델의 이름 또는 경로

    Returns:
        model: 로드된 사전 훈련된 모델
        tokenizer: 해당 모델의 토크나이저
    """
    # AutoModelForCausalLM과 AutoTokenizer를 사용하여 모델과 토크나이저를 로드합니다.
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 패딩 토큰 설정: 패딩 토큰이 없는 경우 eos_token을 사용합니다.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# LoRA 설정을 구성하는 함수
def create_lora_config():
    """
    LoRA 설정 객체를 생성하고 출력합니다.

    Returns:
        lora_config: 생성된 LoRA 설정 객체
    """
    # TODO: LoRA 설정을 생성하세요
    # LoraConfig 클래스를 사용하여 LoRA 설정 객체를 생성합니다.
    # r은 LoRA의 랭크(rank)로, 저랭크 행렬의 차원을 나타냅니다.
    # r 값이 작을수록 파라미터 효율성이 높아지지만, 너무 작으면 모델의 표현력이 감소할 수 있습니다.
    # 일반적으로 성능과 효율성의 균형을 맞추기 위해 r을 8로 설정합니다.
    # lora_alpha는 스케일링 파라미터로, 학습 안정성을 위해 사용되며 보통 r의 몇 배로 설정합니다.
    # 여기서는 r의 4배인 32로 설정하여 학습을 안정화합니다.
    # target_modules는 LoRA를 적용할 모델의 특정 모듈을 지정하며, GPT-2 모델의 어텐션 모듈인 "c_attn"에 적용합니다.
    # LoRA는 linear layer만을 학습할 수 있는데, GPT에서 사용하는 Linear layer중 하나인 'c_attn' 모듈에 대해
    # 학습을 진행해 보겠습니다.
    # lora_dropout은 드롭아웃 확률로, 과적합을 방지하기 위해 일반적으로 0.1로 설정합니다.
    # LoraConfig 클래스 참고: https://huggingface.co/docs/peft/package_reference/lora
    lora_config = LoraConfig(
        r=8,  # LoRA의 랭크(rank)
        lora_alpha=32,  # 스케일링 파라미터
        target_modules=["c_attn"],  # GPT-2 모델의 어텐션 모듈
        lora_dropout=0.1,  # 드롭아웃 확률
    )

    print("LoRA 설정:")
    print(lora_config)

    return lora_config

# 모델에 LoRA를 적용하는 함수
def apply_lora(model, lora_config):
    """
    PEFT의 get_peft_model 함수를 사용하여 모델에 LoRA 설정을 적용합니다.

    Parameters:
        model: 사전 훈련된 모델
        lora_config: LoRA 설정 객체

    Returns:
        model: LoRA가 적용된 모델
    """
    # TODO: 모델에 LoRA를 적용하세요
    # get_peft_model 함수를 사용하여 모델에 LoRA를 적용합니다.
    # - get_peft_model 함수는 두 개의 주요 매개변수를 필요로 합니다:
    #   1. model: 원래의 언어 모델(예: GPT-2, Llama 등)을 나타내며, LoRA가 적용될 대상입니다.
    #      이 모델은 사전 학습된 파라미터를 가지고 있으며, LoRA를 통해 일부 파라미터를 학습 가능하게 만듭니다.
    #   2. lora_config: LoRA 설정 객체로, LoRA를 적용할 모듈, 레이어의 개수, 랭크 등의 설정 정보를 포함합니다.
    #      이를 통해 모델의 특정 모듈에 LoRA를 적용하여 파라미터 수를 줄이면서도 학습 성능을 높일 수 있습니다.
    # - LoRA(Low-Rank Adaptation)는 원래 모델의 파라미터를 수정하지 않고도, 적은 학습 가능한 파라미터를 추가하여 미세 조정할 수 있게 합니다.
    #   이는 메모리 사용량을 줄이고 학습 속도를 개선하는 데 유리합니다.
    # get_peft_model 함수 참고: https://huggingface.co/docs/peft/package_reference/peft_model#peft.get_peft_model
    model = get_peft_model(model, lora_config)
    return model

# 학습 가능한 파라미터 수를 출력하는 함수
def print_trainable_parameters(model):
    """
    모델의 학습 가능한 파라미터 수를 계산하고 출력합니다.

    Parameters:
        model: 분석할 모델
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent = 100 * trainable_params / total_params
    print(f"전체 파라미터 수: {total_params}")
    print(f"학습 가능한 파라미터 수: {trainable_params} ({percent:.2f}% of total)")

def main():
    # 모델 이름 설정
    model_name = "gpt2"  # GPT-2 모델 사용

    # 모델과 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(model_name)

    # LoRA 적용 전 학습 가능한 파라미터 수 출력
    print("LoRA 적용 전:")
    print_trainable_parameters(model)

    # LoRA 설정 생성
    lora_config = create_lora_config()

    # 모델에 LoRA 적용
    model = apply_lora(model, lora_config)

    # LoRA 적용 후 학습 가능한 파라미터 수 출력
    print("\nLoRA 적용 후:")
    print_trainable_parameters(model)

    # 모델이 LoRA 적용으로 미세 조정 준비가 완료됨
    print("\n모델이 LoRA 적용으로 미세 조정할 준비가 되었습니다.")

if __name__ == "__main__":
    main()
