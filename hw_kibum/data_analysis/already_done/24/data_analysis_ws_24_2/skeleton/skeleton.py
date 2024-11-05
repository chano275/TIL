import random
import numpy as np
import os
import torch
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
dataset = load_dataset(____________________})

# 2. 모델과 토크나이저 로드
# 사전 학습된 KoGPT2 모델과 해당 토크나이저를 로드합니다.
# KoGPT2는 한국어 처리에 최적화된 GPT-2 모델입니다.
# 참고: https://github.com/SKT-AI/KoGPT2
model_name = "skt/kogpt2-base-v2"
tokenizer = PreTrainedTokenizerFast.____________________(model_name)
model = GPT2LMHeadModel.____________________(model_name)

# 3. 토크나이저에 새로운 패드 토큰 추가
# GPT-2 모델은 기본적으로 패드 토큰을 사용하지 않으므로, 새로운 패드 토큰을 추가합니다.
# 이는 배치 처리 시 시퀀스 길이를 맞추기 위해 필요합니다. 이를 위해 add_special_tokens 함수를 사용합니다.
# 참고: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.add_special_tokens
tokenizer.____________________({'pad_token': '<pad>'})

# 4. 모델의 토큰 임베딩 크기 조정
# 토크나이저에 패드 토큰이 추가되었으므로, 모델의 임베딩 층을 새로운 토크나이저 크기에 맞게 조정합니다.
# 이는 모델이 새로운 토큰을 인식할 수 있도록 하기 위함입니다. resize_token_embeddings 함수를 사용합니다.
# 참고: https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
model.____________________(len(tokenizer))

# 5. 모델 설정 업데이트
# 모델의 `pad_token_id`와 `vocab_size`를 토크나이저에 맞게 업데이트하여 일관성을 유지합니다.
# 이는 패딩을 올바르게 처리하고, 모델의 어휘 크기를 정확히 반영하기 위함입니다.
# model.config.___를 통해서 해당 모델의 파라미터를 변경할 수 있습니다.
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
# 위에서 정의한 토큰화 함수를 dataset.map에 적용하고, '제목' 열을 제거합니다.
# `batched=True`는 한 번에 여러 샘플을 처리하여 속도를 향상시킵니다.
# 참고: https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.map
tokenized_datasets = _____________________________

# 토큰화 확인
# 첫 번째 샘플을 출력하여 토큰화가 제대로 되었는지 확인합니다.
print(tokenized_datasets['train'][0])

# 최대 input ID 확인
# 데이터셋 내 모든 샘플의 input_ids 중 최대 값을 찾아 모델의 vocab_size와 비교합니다.
max_input_id = max([max(_____________) for ex in tokenized_datasets['train']])
print(f"Max input ID: {max_input_id}")
print(f"Model vocab size: {model.config.vocab_size}")

# input_ids가 모델의 vocab_size를 초과하지 않는지 확인
assert max_input_id < model.config.vocab_size, "Some input_ids exceed the model's vocabulary size."
