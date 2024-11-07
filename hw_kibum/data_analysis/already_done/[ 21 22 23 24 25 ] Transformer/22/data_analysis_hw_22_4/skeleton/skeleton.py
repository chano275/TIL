import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. 사전 학습된 BERT 토크나이저와 분류 모델 불러오기
# 'bert-base-uncased' 모델을 사용하여 토크나이저와 분류 모델을 불러옵니다.
# AutoTokenizer.from_pretrained 함수를 사용하여 사전 학습된 BERT 토크나이저를 불러옵니다.
# 참고: https://huggingface.co/transformers/model_doc/auto.html#transformers.AutoTokenizer.from_pretrained
# AutoModelForSequenceClassification.from_pretrained 함수를 사용하여 사전 학습된 BERT 분류 모델을 불러옵니다.
# 참고: https://huggingface.co/transformers/model_doc/auto.html#transformers.AutoModelForSequenceClassification.from_pretrained
"""
경고 메시지 설명: 'bert-base-uncased' 모델은 일반적인 언어 모델로 사전 학습되었으며, 분류 작업에 맞춰 훈련된 모델이 아닙니다.
이 때문에 분류 레이어(classifier layer)가 초기화되지 않았다는 경고가 발생할 수 있습니다.
해당 경고는 모델이 분류 작업에 대한 학습이 필요하다는 의미이지만, 현재는 모델의 예측 값(로짓)을 확인하는 것이 목적이므로 무시해도 괜찮습니다.
만약 해당 모델을 실제로 텍스트 분류 작업에 사용하려면 추가적인 미세 조정(fine-tuning)이 필요합니다.
"""
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# 2. 문장 정의
# 과제1에서 사용한 동일한 문장을 정의합니다.
sentences = [
    "BERT is a powerful model for natural language processing.",
    "Transformers have revolutionized NLP."
]

for sentence in sentences:
    # 3. 문장 토크나이징
    # tokenizer 함수를 사용하여 문장을 토큰화하고 PyTorch 텐서로 변환합니다.
    # return_tensors='pt'를 사용하여 텐서로 변환된 결과를 반환합니다.
    # 참고: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
    inputs = tokenizer(sentence, return_tensors='pt')

    # 4. 모델에 입력하여 예측 수행
    # torch.no_grad()를 사용해 그래디언트 계산을 생략합니다.
    # 참고: https://pytorch.org/docs/stable/generated/torch.no_grad.html
    # 침고: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward
    with torch.no_grad():
        outputs = model(**inputs)

    # 5. 모델의 예측 결과(로짓) 출력
    # BERT 모델이 예측한 로짓 값을 출력합니다.
    # 로짓 값은 확률이 아닌 미가공 예측 값이므로 Softmax 등의 후처리로 확률로 변환할 수 있습니다.
    print(f"Sentence: {sentence}")
    print(f"Model output (logits): {outputs.logits}")
    print()
