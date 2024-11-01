from transformers import GPT2Tokenizer

# 1. GPT-2 토크나이저 불러오기
# 사전 학습된 GPT-2 토크나이저를 사용하여 텍스트를 처리합니다.
# GPT2Tokenizer.from_pretrained() 함수를 사용하여 GPT-2 토크나이저를 불러옵니다.
# 참고: https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

# 2. 입력 문장 정의
sentence = "GPT models are effective in generating human-like text."

# 3. 문장 토크나이징 및 토큰 수 확인
# tokenizer.tokenize() 함수를 사용하여 문장을 토큰화하고, 토큰 수를 확인합니다.
# 참고: https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.tokenize
# 힌트: sentence를 입력으로 하여 토큰화하세요.
tokens = tokenizer.tokenize(sentence)
print(f"Tokens: {tokens}")

# len() 함수를 사용하여 토큰의 개수를 계산하고 출력합니다.
print(f"Number of tokens: {len(tokens)}")
