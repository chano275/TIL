import numpy as np
import os, torch, random
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

# 시드 고정 (Reproducibility)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# CUDA 사용 가능 여부 확인
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

# 1. 데이터셋 로드
# 'datasets' 라이브러리를 사용하여 CSV 파일에서 데이터를 로드합니다.
# 'train' 키에 해당하는 데이터는 './traffic_news_title.csv' 파일에서 불러옵니다.
# 참고: load_dataset("csv", data_files="my_file.csv")와 같은 형식으로 사용할 수 있습니다.
dataset = load_dataset('csv', data_files={'train': '../data/traffic_news_title.csv'})

# 2. 모델과 토크나이저 로드
# 사전 학습된 KoGPT2 모델과 해당 토크나이저를 로드합니다.
# KoGPT2는 한국어 처리에 최적화된 GPT-2 모델입니다.
# 참고: https://github.com/SKT-AI/KoGPT2
model_name = "skt/kogpt2-base-v2"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

print(tokenizer.tokenize("안녕하세요. 한국어 GPT-2 입니다."))

# 3. 토크나이저에 새로운 패드 토큰 추가
# GPT-2 모델은 기본적으로 패드 토큰을 사용하지 않으므로, 새로운 패드 토큰을 추가합니다.
# 이는 배치 처리 시 시퀀스 길이를 맞추기 위해 필요합니다.
# 참고: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.add_special_tokens
tokenizer.add_special_tokens({'pad_token': '<pad>'})

# 4. 모델의 토큰 임베딩 크기 조정
# 토크나이저에 패드 토큰이 추가되었으므로, 모델의 임베딩 층을 새로운 토크나이저 크기에 맞게 조정합니다.
# 이는 모델이 새로운 토큰을 인식할 수 있도록 하기 위함입니다.
# 참고: https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
model.resize_token_embeddings(len(tokenizer))

# 5. 모델 설정 업데이트
# 모델의 `pad_token_id`와 `vocab_size`를 토크나이저에 맞게 업데이트하여 일관성을 유지합니다.
# 이는 패딩을 올바르게 처리하고, 모델의 어휘 크기를 정확히 반영하기 위함입니다.
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = len(tokenizer)

# 설정 확인
print(f"Tokenizer vocab size: {len(tokenizer)}")          # 예: 51200
print(f"Model vocab size: {model.config.vocab_size}")     # 예: 51200
print(f"Pad token id: {tokenizer.pad_token_id}")          # 예: 3

# 패드 토큰 ID가 모델의 vocab 크기 내에 있는지 확인
assert tokenizer.pad_token_id < model.config.vocab_size, "pad_token_id is out of range."

# 6. 토큰화 함수 정의
# 데이터셋의 '제목' 열을 토크나이즈하여 모델 입력 형식에 맞게 변환하는 함수입니다.
# 이 함수는 각 텍스트를 토큰화하고, 패딩과 트렁케이션을 적용합니다.
# 참고: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
def tokenize_function(examples):
    return tokenizer(
        examples["제목"],                # '제목' 열의 텍스트 데이터를 입력으로 받습니다.
        padding="max_length",            # 모든 입력을 동일한 길이로 패딩합니다.
        truncation=True,                 # 최대 길이를 초과하는 경우 자릅니다.
        max_length=32,                   # 최대 토큰 길이를 32로 설정합니다.
    )

# 7. 토큰화 적용
# 위에서 정의한 토큰화 함수를 데이터셋에 적용하고, '제목' 열을 제거합니다.
# `batched=True`는 한 번에 여러 샘플을 처리하여 속도를 향상시킵니다.
# 참고: https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.map
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["제목"])

# 토큰화 확인
# 첫 번째 샘플을 출력하여 토큰화가 제대로 되었는지 확인합니다.
print(tokenized_datasets['train'][0])

# 최대 input ID 확인
# 데이터셋 내 모든 샘플의 input_ids 중 최대 값을 찾아 모델의 vocab_size와 비교합니다.
max_input_id = max([max(ex['input_ids']) for ex in tokenized_datasets['train']])
print(f"Max input ID: {max_input_id}")
print(f"Model vocab size: {model.config.vocab_size}")

# input_ids가 모델의 vocab_size를 초과하지 않는지 확인
assert max_input_id < model.config.vocab_size, "Some input_ids exceed the model's vocabulary size."


# 8. 데이터 콜레이터 정의
# 언어 모델링을 위한 데이터 콜레이터를 정의합니다.
# `mlm=False`는 마스킹을 사용하지 않는다는 것을 의미하며, 이는 GPT-2가 언어 모델링을 위한 causal LM이기 때문입니다.
# 참고: https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# 9. 훈련 하이퍼파라미터 설정
# 모델 훈련에 필요한 여러 하이퍼파라미터와 설정을 정의합니다.
# `TrainingArguments`는 훈련 과정 전반에 걸친 설정을 관리합니다.
# 참고: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",               # 모델 체크포인트와 결과를 저장할 디렉터리
    overwrite_output_dir=True,            # 기존 출력 디렉터리를 덮어쓸지 여부
    num_train_epochs=3,                   # 전체 데이터셋을 몇 번 반복할지 설정 (에포크 수)
    per_device_train_batch_size=2,        # 훈련 시 각 디바이스(예: GPU)당 배치 크기
    learning_rate=5e-5,                   # 학습률 설정
)

# 10. Trainer 초기화
# `Trainer` 객체를 초기화하여 모델, 훈련 인수, 데이터셋, 데이터 콜레이터 등을 설정합니다.
# `Trainer`는 훈련 루프, 평가, 예측 등을 간편하게 수행할 수 있도록 도와줍니다.
# 참고: https://huggingface.co/docs/transformers/main_classes/trainer
trainer = Trainer(
    model=model,                                      # 훈련할 모델
    args=training_args,                               # 훈련 인수
    train_dataset=tokenized_datasets['train'],        # 훈련 데이터셋
    data_collator=data_collator,                      # 데이터 콜레이터
)

print(f"Trainer의 모델 : {trainer.model}")

# 11. 모델 훈련
# `Trainer` 객체를 사용하여 모델을 훈련시킵니다.
# 훈련 과정 동안 지정한 설정에 따라 모델이 업데이트됩니다.
trainer.train()

# 12. 모델과 토크나이저 저장
# 훈련된 모델과 토크나이저를 지정된 디렉터리에 저장합니다.
# 이는 나중에 모델을 불러와서 사용할 수 있도록 합니다.
# 참고: https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.save_pretrained
trainer.save_model("./results/gpt2-title-generator")
tokenizer.save_pretrained("./results/gpt2-title-generator")

# 13. 추론을 위한 모델과 토크나이저 로드
# 저장된 모델과 토크나이저를 로드하여 추론에 사용합니다.
# 이는 훈련된 모델을 사용하여 새로운 데이터를 생성할 때 필요합니다.
model = GPT2LMHeadModel.from_pretrained('./results/gpt2-title-generator')
tokenizer = PreTrainedTokenizerFast.from_pretrained('./results/gpt2-title-generator')

print(f"파인튜닝한 Trainer의 모델 : {trainer.model}")


# 제목 생성 함수 정의
# 입력 텍스트를 기반으로 새로운 제목을 생성하는 함수입니다.
# `generate` 메소드를 사용하여 텍스트를 생성하며, 다양한 파라미터를 통해 생성 방식을 제어할 수 있습니다.
# 참고: https://huggingface.co/docs/transformers/main_classes/model#transformers.GenerationMixin.generate
def generate_title(input_text, max_length=32):
    # 입력 텍스트를 토크나이즈하고 모델 디바이스로 이동시킵니다.
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(model.device)

    # 텍스트 생성
    gen_ids = model.generate(
        input_ids,
        max_length=max_length,                # 생성할 텍스트의 최대 길이
        num_return_sequences=1,               # 생성할 시퀀스 수
        no_repeat_ngram_size=2,               # n-그램 반복 방지 (여기서는 Bi-gram 이상 반복 금지로 지정)
        repetition_penalty=2.0,               # 반복 억제 페널티 (값이 클수록 반복을 덜함)
        eos_token_id=tokenizer.eos_token_id,  # 종료 토큰 ID
        pad_token_id=tokenizer.pad_token_id,  # 패드 토큰 ID
        top_k=50,                             # 상위 K개의 토큰 중에서 샘플링
        top_p=0.95,                           # 누적 확률이 top_p 이상이 되는 토큰 중에서 샘플링
        temperature=1.9,                      # 샘플링 온도 (값이 높을수록 다양성이 높아짐)
    )

    # 생성된 텍스트 디코딩
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return gen_text

# 제목 생성 예시
# 입력 텍스트를 기반으로 제목을 생성하고 출력합니다.
input_text = "서울시 지하철 요금"
generated_title = generate_title(input_text, max_length=32)
print("Generated Title:", generated_title)
