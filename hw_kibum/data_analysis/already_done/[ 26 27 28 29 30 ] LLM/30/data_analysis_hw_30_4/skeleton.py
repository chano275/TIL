import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(tokenizer):
    """
    금융 데이터셋을 로드하고 토큰화합니다.

    Returns:
        input_ids: 토큰화된 입력 IDs
        attention_mask: 어텐션 마스크
    """
    # 예시 금융 데이터
    texts = [
        "2023년 한국의 경제 성장률은 예상보다 높을 것으로 전망됩니다.",
        "금융 시장은 최근의 변동성으로 인해 불안정한 상태입니다.",
        "인플레이션 상승으로 중앙은행은 금리 인상을 고려하고 있습니다.",
    ]

    # 텍스트를 토큰화하고 텐서로 변환
    encoding = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    return input_ids, attention_mask

# 모델과 토크나이저 로드 및 LoRA 적용 함수
def load_model_and_apply_lora(model_name: str):
    """
    모델과 토크나이저를 로드하고 LoRA를 적용합니다.

    Parameters:
        model_name (str): 사전 훈련된 모델의 이름

    Returns:
        model: LoRA가 적용된 모델
        tokenizer: 해당 모델의 토크나이저
    """
    # TODO: 모델과 토크나이저를 로드하세요
    # AutoModelForCausalLM.from_pretrained()와 AutoTokenizer.from_pretrained()를 사용하여
    # 모델과 토크나이저를 로드합니다. 이때, 함수의 인자로 받은 model_name 변수를 사용합니다.
    # AutoModelForCausalLM 참고: https://huggingface.co/transformers/model_doc/gpt2.html#aut
    # AutoTokenizer 참고: https://huggingface.co/transformers/model_doc/gpt2.html#autotokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # TODO: LoRA 설정을 구성하고 모델에 적용하세요
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
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["c_attn"], lora_dropout=0.1)

    #TODO: get_peft_model() 함수를 사용하여 모델에 LoRA를 적용하세요.
    model = get_peft_model(model, lora_config)
    return model, tokenizer

# 모델 학습 함수
def train_model(model, input_ids, attention_mask):
    """
    간단한 학습 루프를 통해 모델을 미세 조정합니다.

    Parameters:
        model: 학습할 모델
        input_ids: 토큰화된 입력 IDs
        attention_mask: 어텐션 마스크
    """
    # 모델을 학습 모드로 전환
    model.train()

    # 옵티마이저 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 손실 함수 정의 (모델에 내장되어 있음)
    epochs = 3
    for epoch in range(epochs):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        # 역전파
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"에폭 {epoch + 1}/{epochs}, 손실: {loss.item():.4f}")

def main():
    # 모델 이름 설정
    model_name = "gpt2"

    # 모델과 토크나이저 로드 및 LoRA 적용
    model, tokenizer = load_model_and_apply_lora(model_name)

    # 데이터 로드 및 전처리
    input_ids, attention_mask = load_and_preprocess_data(tokenizer)

    # 모델 학습
    train_model(model, input_ids, attention_mask)

    # 모델 저장
    model.save_pretrained("finetuned_model")
    tokenizer.save_pretrained("finetuned_model")

    print("\n모델 학습 및 저장이 완료되었습니다.")

if __name__ == "__main__":
    main()
