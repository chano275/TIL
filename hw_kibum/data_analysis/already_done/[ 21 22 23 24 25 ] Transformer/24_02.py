import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# 1. GPT-2 모델 및 토크나이저 불러오기 - 사전 학습된 GPT-2 토크나이저와 모델을 불러옵니다. - 사용한 모델: 'gpt2'
# AutoTokenizer.from_pretrained()함수를 사용하여 GPT-2 토크나이저를 불러옵니다. - 참고: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer
# AutoModelForCausalLM.from_pretrained()함수를 사용하여 GPT-2 모델을 불러옵니다. - 참고: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

# 2. 입력 문장 정의 - 이어서 텍스트를 생성할 초기 문장을 정의합니다.
prompt = "Generative AI is transforming industries. "

# 3. 문장 토크나이징 및 input_ids 변환 - tokenizer() 함수를 사용하여 문장을 토큰화하고 텐서로 변환합니다.
inputs = tokenizer(prompt, return_tensors='pt')

# 4. 모델을 사용해 텍스트 생성 - generate() 함수를 사용해 입력된 문장에 이어 텍스트를 생성합니다.
# max_length=30으로 최대 길이를 30으로 제한하고, num_return_sequences=1로 설정하여 1개의 텍스트만 생성합니다.
# pad_token_id를 지정하지 않고 경고 메시지를 무시한 상태로 텍스트 생성합니다.
# 참고: https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.generate
output = model.generate(inputs['input_ids'], max_length=30, num_return_sequences=1)

# 5. 생성된 텍스트 디코딩 및 출력 - 생성된 텍스트를 디코딩하여 사람이 읽을 수 있는 형식으로 변환합니다.
# skip_special_tokens=True 옵션을 사용하여 특수 토큰을 제거 - 특수 토큰을 제거하는 이유: 생성된 텍스트에서 특수 토큰(<EOS> 등)을 제거하여 읽기 쉽게 만듭니다.
# 참고: https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.decode
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated Text: {generated_text}")




#########################################################################################
# 1. GPT-2 토크나이저 불러오기 - 사전 학습된 GPT-2 토크나이저를 불러옵니다.
# AutoTokenizer.from_pretrained 함수를 사용하여 GPT-2 토크나이저를 불러옵니다.(gpt2-GPT2LMHeadModel(OpenAI GPT-2 model))
# 참고: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# 2. 입력 문장 정의 - 토크나이징할 문장을 정의합니다.
sentence = "I love transformers."

# 3. 문장 토크나이징 - tokenizer 함수를 사용하여 문장을 토큰화하고 PyTorch 텐서로 변환
# return_tensors='pt'를 사용하여 텐서로 변환된 결과를 반환 - 참고: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
inputs = tokenizer(sentence, return_tensors='pt')

# 4. 토크나이징 결과 출력 - 토큰화된 input_ids와 대응되는 단어를 출력 - tokenizer.convert_ids_to_tokens 함수를 사용하여 input_ids를 단어로 변환
# 참고: https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.convert_ids_to_tokens
# inputs는 딕셔너리 형태이며, 'input_ids'라는 키를 통해 토큰화된 ID에 접근 가능 - 이 값은 텐서이므로 첫 번째 차원에 접근해야 합니다.
tokenized_sentence = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# 5. 토큰 및 해당 토큰 ID 출력
for idx, token in enumerate(tokenized_sentence):
    print(f"Token at index {idx}: {token} (Token ID: {inputs['input_ids'][0][idx]})")
