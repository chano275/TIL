import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset
import evaluate  # 평가 메트릭을 계산하기 위한 라이브러리

# 1. 데이터 로드 및 준비
# CSV 파일을 판다스 데이터프레임으로 로드합니다.
df = pd.read_csv('../data/traffic_news_eng.csv')

# 텍스트와 레이블 추출
# '제목' 열에서 텍스트 데이터를, '교통관련' 열에서 레이블 데이터를 리스트 형태로 추출합니다.
texts = df['제목'].tolist()
labels = df['교통관련'].tolist()

# 데이터셋 분할 (train/test split)
# 데이터를 훈련 세트와 테스트 세트로 80:20 비율로 나눕니다.
# random_state=42는 결과의 재현성을 위해 설정합니다.
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Hugging Face의 Dataset 객체로 변환
# 판다스의 딕셔너리 형태로 데이터를 Hugging Face의 Dataset 객체로 변환합니다.
# Dataset 객체는 Hugging Face의 transformers 라이브러리와 호환되며, 효율적인 데이터 처리를 가능하게 합니다.
# 참고: https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset
# Dataset의 from_dict() 함수를 통해서 text와 label로 나눠줄 수 있습니다.
train_dataset = Dataset.___________({'text': train_texts, 'label': train_labels})
test_dataset = Dataset.___________({'text': test_texts, 'label': test_labels})

# 2. 토크나이저 로드
# BERT 사전학습된 모델에서 토크나이저 가져오기 (bert-base-uncased 사용)
# transformers에서 사전에 구현되고 학습된 tokenizer를 단순하게 불러올 수 있습니다.
# from_pretrained() 함수를 사용하고, 아래의 사이트에 공개된 모델을 로드할 수 있습니다.
# https://huggingface.co/models 에서 검색해서 다양한 모델을 사용할 수 있습니다.
tokenizer = BertTokenizer.__________________('bert-base-uncased')


# 3. 데이터 전처리 (토큰화)
# 텍스트 데이터를 BERT 모델이 이해할 수 있는 토큰으로 변환하는 과정을 함수로 정의합니다.
# 참고: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
def tokenize_function(examples):
    return tokenizer(
        examples['text'],              # 텍스트 데이터를 입력으로 받습니다.
        padding='max_length',          # 모든 입력을 동일한 길이로 패딩합니다.
        truncation=True,               # 최대 길이를 초과하는 경우 잘라냅니다.
        max_length=128                 # 최대 토큰 길이를 128로 설정합니다.
    )

# 훈련 데이터와 테스트 데이터에 토큰화 함수를 적용합니다.
# map() 함수를 통해 기존에 tokenize 과정을 map 형태로 처리합니다.
# batched=True는 한 번에 여러 샘플을 처리하여 속도를 향상시킵니다.
# 참고: https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.map
train_dataset = train_dataset.______________________
test_dataset = test_dataset.______________________

# 포맷 지정 (torch 텐서로 변환)
# 모델이 PyTorch를 사용하기 때문에 데이터를 PyTorch 텐서 형식으로 변환합니다.
# 참고: https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.set_format
train_dataset.______________________
test_dataset.______________________

# 4. 모델 로드
# 사전 학습된 BERT 모델을 불러옵니다.
# BertForSequenceClassification은 문장 분류 작업을 위한 BERT 모델입니다.
# num_labels=2는 이진 분류 작업을 의미합니다.
# model도 from_pretrained() 함수를 사용하고, 아래의 사이트에 공개된 모델을 로드할 수 있습니다.
# 참고: https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification
model = BertForSequenceClassification._________________('bert-base-uncased', ______________)

print(f"model: {model}")