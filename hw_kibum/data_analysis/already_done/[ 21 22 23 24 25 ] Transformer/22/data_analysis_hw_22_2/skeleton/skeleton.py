from transformers import AutoTokenizer

# 1. 사전 학습된 BERT 토크나이저 불러오기
# AutoTokenizer.from_pretrained 함수를 사용하여 사전 학습된 BERT 토크나이저를 불러옵니다.
# 'bert-base-cased' 모델을 사용하여 대소문자를 구분하며 토크나이징합니다.
# AutoTokenizer.from_pretrained 함수 참고: https://huggingface.co/transformers/model_doc/auto.html#autotokenizer
# 사용한 BERT 모델: 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# 2. 토크나이징할 문장 정의
# 토크나이징할 문장을 리스트 형태로 정의합니다.
sentences = [
    "BERT is a powerful model for Natural Language Processing.",
    "Transformers have revolutionized the field of NLP."
]

# 3. 문장 토크나이징 및 출력
# tokenizer.tokenize 함수를 사용하여 문장을 토큰화하고, 결과를 출력합니다.
for sentence in sentences:
    tokenized_output = tokenizer.tokenize(sentence)
    print(f"Tokenized: {tokenized_output}")
