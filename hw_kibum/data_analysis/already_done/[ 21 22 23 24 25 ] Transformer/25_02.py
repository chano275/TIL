import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 1. GPT-2 토크나이저 불러오기
# 사전 학습된 GPT-2 토크나이저를 사용하여 텍스트를 처리합니다.
# GPT2Tokenizer.from_pretrained() 함수를 사용하여 GPT-2 토크나이저를 불러옵니다.
# 참고: https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

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

###########################################################################################

# 1. GPT-2 모델 및 토크나이저 불러오기
# 사전 학습된 GPT-2 모델과 토크나이저를 불러옵니다.
# GPT2Tokenizer.from_pretrained() 함수를 사용하여 GPT-2 토크나이저를 불러옵니다.
# GPT2LMHeadModel.from_pretrained() 함수를 사용하여 GPT-2 모델을 불러옵니다.
# 참고: https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Tokenizer
# 참고: https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 2. 입력 문장 정의
# 이어서 텍스트를 생성할 프롬프트 문장을 정의합니다.
prompt = "The number 7 is important because "

# 3. 문장 토크나이징 및 input_ids 변환
# tokenizer() 함수를 사용하여 문장을 토큰화한 후 텐서로 변환합니다.
# return_tensors 옵션을 'pt'로 설정하여 텐서로 변환된 결과를 반환합니다.
# 참고: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
inputs = tokenizer(prompt, return_tensors='pt')

# 4. 모델을 사용해 텍스트 생성 (특정 조건에 맞춘 텍스트 생성)
# generate() 함수를 사용해 텍스트를 생성합니다.
# 텐서로 변환된 입력 문장을 입력으로 사용합니다. 이때 input_ids를 사용합니다.
# max_length=30을 설정하여 최대 길이를 30으로 제한하세요. num_return_sequences=1로 설정하여 1개의 텍스트만 생성합니다.
output = model.generate(inputs['input_ids'], max_length=30, num_return_sequences=1)

# 5. 생성된 텍스트 디코딩 및 출력
# tokenizer.decode() 함수를 사용해 생성된 텍스트를 디코딩합니다.
# skip_special_tokens=True를 사용해 특수 토큰을 제거하세요.
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated Text: {generated_text}")
