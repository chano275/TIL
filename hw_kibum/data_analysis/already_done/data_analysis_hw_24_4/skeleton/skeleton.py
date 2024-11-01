import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. GPT-2 모델 및 토크나이저 불러오기
# 사전 학습된 GPT-2 토크나이저와 모델을 불러옵니다.
# 사용한 모델: 'gpt2'
# AutoTokenizer.from_pretrained()함수를 사용하여 GPT-2 토크나이저를 불러옵니다.
# 참고: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer
# AutoModelForCausalLM.from_pretrained()함수를 사용하여 GPT-2 모델을 불러옵니다.
# 참고: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

# 2. 입력 문장 정의
# 이어서 텍스트를 생성할 초기 문장을 정의합니다.
prompt = "Generative AI is transforming industries. "

# 3. 문장 토크나이징 및 input_ids 변환
# tokenizer() 함수를 사용하여 문장을 토큰화하고 텐서로 변환합니다.
inputs = tokenizer(prompt, return_tensors='pt')

# 4. 모델을 사용해 텍스트 생성
# generate() 함수를 사용해 입력된 문장에 이어 텍스트를 생성합니다.
# max_length=30으로 최대 길이를 30으로 제한하고, num_return_sequences=1로 설정하여 1개의 텍스트만 생성합니다.
# pad_token_id를 지정하지 않고 경고 메시지를 무시한 상태로 텍스트 생성합니다.
# 참고: https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.generate
output = model.generate(inputs['input_ids'], max_length=30, num_return_sequences=1)

# 5. 생성된 텍스트 디코딩 및 출력
# 생성된 텍스트를 디코딩하여 사람이 읽을 수 있는 형식으로 변환합니다.
# skip_special_tokens=True 옵션을 사용하여 특수 토큰을 제거합니다.
# 특수 토큰을 제거하는 이유: 생성된 텍스트에서 특수 토큰(<EOS> 등)을 제거하여 읽기 쉽게 만듭니다.
# 참고: https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.decode
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated Text: {generated_text}")
