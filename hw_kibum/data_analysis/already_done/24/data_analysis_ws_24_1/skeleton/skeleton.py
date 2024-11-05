import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from datasets import load_dataset

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

print(tokenizer.tokenize("안녕하세요. 한국어 GPT-2 입니다."))