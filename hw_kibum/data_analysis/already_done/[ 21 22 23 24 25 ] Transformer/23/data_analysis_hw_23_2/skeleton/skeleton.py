from transformers import AutoTokenizer

# 1. DistilBERT 토크나이저 불러오기
# AutoTokenizer.from_pretrained 함수 참고: https://huggingface.co/transformers/model_doc/auto.html#autotokenizer
# 사용한 모델: 'distilbert-base-uncased'
# 참고: https://huggingface.co/distilbert-base-uncased
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# 2. 입력 문장 정의
sentences = [
    "Data analysis is crucial for business success.",
    "Artificial intelligence is transforming industries."
]

# 3. 문장 토크나이징 및 input_ids 변환
for sentence in sentences:
    # tokenizer() 함수를 사용하여 문장을 토큰화한 후 텐서로 변환합니다.
    # return_tensors 옵션을 'pt'로 설정하여 텐서로 변환된 결과를 반환합니다.
    # 참고: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
    inputs = tokenizer(sentence, return_tensors='pt')


    print(f"Sentence: {sentence}")
    print(f"Input IDs: {inputs['input_ids']}")
    print(f"Attention Mask: {inputs['attention_mask']}")
