import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. RoBERTa 모델 불러오기
# AutoTokenizer.from_pretrained 함수를 사용하여 RoBERTa 토크나이저를 불러옵니다.
# AutoModelForSequenceClassification.from_pretrained 함수를 사용하여 사전 학습된 RoBERTa 모델을 불러옵니다.
# 참고 링크: https://huggingface.co/transformers/model_doc/auto.html#autotokenizer
# 사용한 RoBERTa 모델: 'roberta-base'
# 참고: https://huggingface.co/roberta-base
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base')

# 2. 입력 문장 정의
# 예측을 위해 토크나이징할 문장을 정의합니다.
sentence = "RoBERTa is an optimized version of BERT."

# 3. 문장 토크나이징
# tokenizer 함수를 사용하여 문장을 토큰화하고 PyTorch 텐서로 변환합니다.
# return_tensors='pt'를 사용하여 텐서로 변환된 결과를 반환합니다.
# 참고: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
inputs = tokenizer(sentence, return_tensors='pt')

# 4. 모델 예측
# torch.no_grad()를 사용해 예측 시 그래디언트 계산을 방지하여 메모리 사용을 줄입니다.
# 모델에 입력 텐서를 전달해 예측 결과를 얻습니다.
with torch.no_grad():
    outputs = model(**inputs)

# 5. 로짓 값 추출
# 모델의 출력 결과인 로짓(logits)을 추출합니다.
# 로짓은 아직 확률 값으로 변환되지 않은 예측 결과입니다.
logits = outputs.logits

# 6. Softmax로 확률값 변환
# torch.softmax() 함수를 사용해 로짓 값을 확률로 변환합니다.
# dim=1은 각 클래스에 대한 확률값을 계산함을 의미합니다.
probabilities = torch.softmax(logits, dim=1)
print(f"Probabilities: {probabilities}")

# 7. 가장 높은 확률을 가진 클래스 출력
# torch.argmax()를 사용해 가장 높은 확률을 가진 클래스를 선택합니다.
predicted_class = torch.argmax(probabilities, dim=1)
print(f"Predicted class: {predicted_class}")
