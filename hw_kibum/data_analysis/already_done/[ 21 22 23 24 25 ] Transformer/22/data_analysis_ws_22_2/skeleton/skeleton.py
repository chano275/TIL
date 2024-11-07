import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from datasets import Dataset

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
