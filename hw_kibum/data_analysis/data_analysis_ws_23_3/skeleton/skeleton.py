import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset

# 1. 데이터 로드 및 준비
# CSV 파일을 판다스 데이터프레임으로 로드합니다.
df = pd.read_csv('../data/traffic_news.csv')

# 텍스트와 레이블 추출
# '제목' 열에서 텍스트 데이터를, '교통관련' 열에서 레이블 데이터를 리스트 형태로 추출합니다.
texts = df['제목'].tolist()
labels = df['교통관련'].tolist()

# 2. HuggingFace Dataset 객체로 변환
# Hugging Face의 Dataset 객체로 변환
# 판다스의 딕셔너리 형태로 데이터를 Hugging Face의 Dataset 객체로 변환합니다.
# Dataset 객체는 Hugging Face의 transformers 라이브러리와 호환되며, 효율적인 데이터 처리를 가능하게 합니다.
dataset = Dataset.___________({'text': texts, 'label': labels})

# 3. 토크나이저 로드
# AutoTokenizer를 사용하여 RoBERTa 모델의 토크나이저 로드 (한국어를 다루기 위해 한국어 기반으로 학습한 'klue/roberta-base' 모델 사용)
tokenizer = AutoTokenizer.___________('klue/roberta-base')

# 4. 토크나이저 적용을 위한 함수 정의
def tokenize_data(examples):
    # 텍스트 데이터를 BERT 모델이 이해할 수 있는 토큰으로 변환하는 과정을 함수로 정의합니다.
    # 참고: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
    # text 데이터를 토큰화하고 'input_ids', 'attention_mask' 반환
    return tokenizer(
        examples['text'],              # 텍스트 데이터를 입력으로 받습니다.
        padding='max_length',          # 모든 입력을 동일한 길이로 패딩합니다.
        truncation=True,               # 최대 길이를 초과하는 경우 잘라냅니다.
        max_length=128                 # 최대 토큰 길이를 128로 설정합니다.
    )


# 5. 데이터셋에 토크나이저 적용 및 '제목' 열 제거
# map() 메소드를 사용해 데이터셋의 모든 텍스트에 토크나이저를 적용하며, '제목' 열을 제거합니다.
# batched=True는 한 번에 여러 샘플을 처리하여 속도를 향상시킵니다.
# remove_columns은 불필요한 컬럼을 제거합니다.
# 참고: https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.map
tokenized_dataset = ______________________

# 6. Train/Validation 데이터셋 나누기
# train_test_split 메소드를 사용하여 데이터셋을 80:20 비율로 훈련 및 테스트 세트로 나눕니다.
train_test_data = tokenized_dataset.train_test_split(test_size=0.2,, random_state=42)
train_dataset = train_test_data['train']
test_dataset = train_test_data['test']


# 7. 데이터를 배치로 묶을 때 패딩을 자동으로 추가하는 DataCollator
# 모델에 입력되는 텍스트 시퀀스들은 서로 길이가 다를 수 있습니다.
# BERT, RoBERTa와 같은 트랜스포머 모델은 배치 입력에서 시퀀스 길이가 동일해야 하므로,짧은 시퀀스에 패딩을 추가해 길이를 맞추는 작업이 필요합니다.
# DataCollatorWithPadding은 데이터를 배치로 묶을 때 자동으로 패딩을 추가해주는 역할을 합니다.
# 즉, 각 배치마다 입력 시퀀스 중 가장 긴 시퀀스의 길이에 맞춰 나머지 시퀀스들에 패딩을 추가하여 배치를 구성합니다.
# 참고: https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorWithPadding
data_collator = ______________________(tokenizer=tokenizer)

# 8. 모델 로드
# 사전 학습된 BERT 모델을 불러옵니다.
# AutoModelForSequenceClassification을 사용해 사전 학습된 RoBERTa 기반의 문장 분류 모델을 로드합니다.
# 'klue/roberta-base' 모델은 한국어 데이터 처리를 위해 사용되며, num_labels=2는 이진 분류 작업을 의미합니다.
# model도 from_pretrained() 함수를 사용하고, 아래의 사이트에 공개된 모델을 로드할 수 있습니다.
# 참고: https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification
model = AutoModelForSequenceClassification._________________("klue/roberta-base", ______________)

# 자동으로 RobertaForSequenceClassification 모델을 불러옵니다.
print(f"model: {model}")