
import os
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast


# 1. 토크나이저 로드
def load_tokenizer(model_name='skt/kogpt2-base-v2'):
    """
    사전 학습된 GPT-2 토크나이저를 로드하고, 필요시 EOS 토큰과 패딩 토큰을 설정하는 함수
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    
    # 참고: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.add_special_tokens
    # EOS 토큰이 설정되지 않은 경우, 새로운 EOS 토큰을 추가
    # TODO: <eos> 형태로 'eos_token'을 추가해주세요.
    if tokenizer.eos_token is None:
        tokenizer._____________________
    
    # 패딩 토큰이 설정되지 않은 경우, EOS 토큰을 패딩 토큰으로 설정
    # TODO: 'pad_token'으로 eos_token을 매핑해주세요.
    if tokenizer.pad_token is None:
        tokenizer._____________________
    
    return tokenizer

# 2. 데이터셋 로드 및 전처리
def load_dataset_and_preprocess(file_path='traffic_data.csv', model_name='skt/kogpt2-base-v2', max_length=128):
    """
    CSV 파일로부터 데이터를 로드하고, 토크나이저를 통해 텍스트 데이터를 전처리하는 함수
    :param file_path: 데이터 파일 경로
    :param model_name: 사용할 GPT-2 모델 이름
    :param max_length: 토큰화할 텍스트의 최대 길이
    """
    tokenizer = _____________________(model_name=model_name)  # TODO: 이전에 만든 토크나이저 로드
    data = _____________________('csv', data_files=file_path, split='train')  # TODO: datasets.load_dataset를 통해 train데이터로 로드
    if 'text' not in data.column_names:
        raise ValueError("CSV 파일에 'text' 열이 존재하지 않습니다.")
    
    # 텍스트 데이터를 토큰화하는 함수
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    # 참고: https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.map
    # TODO: data를 map함수를 이용해서 tokenized_dataset으로 표현해주세요.
    tokenized_dataset = data._____________________  # 데이터셋을 토큰화
    
    # 'input_ids'와 동일하게 'labels' 컬럼을 설정 (모델 학습에 사용)
    tokenized_dataset = tokenized_dataset.map(lambda x: {'labels': x['input_ids']}, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])  # 데이터셋 포맷 설정
    
    return tokenized_dataset


def main():
    file_path = 'traffic_data.csv'
    if not os.path.exists(file_path):
        print(f"{file_path} 파일이 존재하지 않습니다. 데이터를 먼저 준비해주세요.")
        return

    model_name='skt/kogpt2-base-v2'
    tokenized_dataset = load_dataset_and_preprocess(file_path, model_name, max_length=128)
    print(f"tokenized_dataset: {tokenized_dataset}")
    print("데이터셋이 성공적으로 로드되고 전처리되었습니다.")

if __name__ == "__main__":
    main()
