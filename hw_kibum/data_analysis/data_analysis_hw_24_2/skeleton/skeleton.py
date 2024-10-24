from transformers import AutoTokenizer

# 1. GPT-2 토크나이저 불러오기
# 사전 학습된 GPT-2 토크나이저를 불러옵니다.
# AutoTokenizer.from_pretrained 함수를 사용하여 GPT-2 토크나이저를 불러옵니다.(gpt2-GPT2LMHeadModel(OpenAI GPT-2 model))
# 참고: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# 2. 입력 문장 정의
# 토크나이징할 문장을 정의합니다.
sentence = "I love transformers."

# 3. 문장 토크나이징
# tokenizer 함수를 사용하여 문장을 토큰화하고 PyTorch 텐서로 변환합니다.
# return_tensors='pt'를 사용하여 텐서로 변환된 결과를 반환합니다.
# 참고: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
inputs = tokenizer(sentence, return_tensors='pt')

# 4. 토크나이징 결과 출력
# 토큰화된 input_ids와 대응되는 단어를 출력합니다.
# tokenizer.convert_ids_to_tokens 함수를 사용하여 input_ids를 단어로 변환합니다.
# 참고: https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.convert_ids_to_tokens
# inputs는 딕셔너리 형태이며, 'input_ids'라는 키를 통해 토큰화된 ID에 접근할 수 있습니다.
# 이 값은 텐서이므로 첫 번째 차원에 접근해야 합니다.
tokenized_sentence = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# 5. 토큰 및 해당 토큰 ID 출력
for idx, token in enumerate(tokenized_sentence):
    print(f"Token at index {idx}: {token} (Token ID: {inputs['input_ids'][0][idx]})")
