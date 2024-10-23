'''
이번 실습에서는 허깅페이스를 통해 Roberta를 기반으로 하는 모델과 기타 모델을 추상화해서 다루는 방식을 전체적으로 알아봅니다.
'''
from transformers import AutoTokenizer

# BERT 사전학습된 모델에서 토크나이저 가져오기 (bert-base-uncased 사용)
# transformers에서 사전에 구현되고 학습된 tokenizer를 단순하게 불러올 수 있습니다.
# from_pretrained() 함수를 사용하고, 아래의 사이트에 공개된 모델을 로드할 수 있습니다.
# https://huggingface.co/models 에서 검색해서 다양한 모델을 사용할 수 있습니다.

# BERT 사전 학습된 모델의 토크나이저 로드 ('bert-base-uncased')
# AutoTokenizer.from_pretrained()를 사용하여 토크나이저를 불러올 수 있으며,
# 모델명만 바꾸면 다양한 모델의 토크나이저를 불러올 수 있습니다.
tokenizer_bert = ____________________.from_pretrained('bert-base-uncased')

# RoBERTa 사전 학습된 모델의 토크나이저 로드 ('roberta-base')
# 동일한 AutoTokenizer.from_pretrained() 메소드를 사용하여 RoBERTa 토크나이저도 불러올 수 있습니다.
tokenizer_roberta = ____________________.from_pretrained('roberta-base')

# 데이터 예시 (교통 데이터의 텍스트)
sample_text = "The traffic situation in Seoul is getting worse due to heavy rain."

# 텍스트를 BERT 기반의 모델 기반의 토크나이저를 이용해 토큰화
# 모델의 토크나이저를 로드합니다.
# 'bert-base-uncased'와 같이 공유한 모델을 허깅페이스에서 사이트에서 모델을 검색해서 불러올 수 있습니다.
# 참고: https://huggingface.co/docs/transformers/main_classes/tokenizer

# tokenize() 함수를 통해서 token으로 나눌 수 있습니다.
# convert_tokens_to_ids를 통해 token들에 해당하는 id를 불러올 수 있습니다.
# BERT 토크나이저를 사용하여 텍스트 토큰화
tokens_bert = tokenizer_bert.____________________(sample_text)
# 토큰을 ID로 변환 (모델의 어휘집에 따라 변환됨)
token_ids_bert = tokenizer_bert.____________________(tokens_bert)

# RoBERTa 토크나이저를 사용하여 텍스트 토큰화
tokens_roberta = tokenizer_roberta.____________________(sample_text)
# RoBERTa 토큰을 ID로 변환
token_ids_roberta = tokenizer_roberta.____________________(tokens_roberta)

print(f"Tokens(BERT): {tokens_bert}")
print(f"Token IDs(BERT): {token_ids_bert}")

print(f"Tokens(RoBERTa): {tokens_roberta}")
print(f"Token IDs(RoBERTa): {token_ids_roberta}")
