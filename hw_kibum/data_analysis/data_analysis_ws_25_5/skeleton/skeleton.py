import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import re

# 시드 고정
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 시드를 고정하여 재현 가능하게 설정
set_seed(42)


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

# 3. 데이터로더 생성
def create_dataloader(dataset, batch_size=2, shuffle=True):
    """
    데이터셋을 기반으로 데이터로더를 생성하는 함수
    :param dataset: HuggingFace Dataset 객체
    :param batch_size: 배치 크기 (기본값 2)
    :param shuffle: 데이터 섞기 여부 (기본값 True)
    """
    dataloader = DataLoader(_____________________)  # TODO: 데이터로더 생성을 위해 받아온 인자값을 통해 dataloader를 만들어주세요.
    return dataloader

# 4. GPT-2 모델 로드 및 설정
def load_model(model_name='skt/kogpt2-base-v2', tokenizer=None):
    """
    사전 학습된 GPT-2 모델을 로드하고, 토크나이저의 어휘 크기에 맞게 조정하는 함수
    :param model_name: 사용할 모델 이름
    :param tokenizer: 토크나이저 객체
    """
    model = GPT2LMHeadModel.from_pretrained(model_name)  # 모델 로드
    
    # 패딩 토큰을 추가한 경우, 임베딩 레이어 크기 조정
    # 참고: https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
    # 모델이 새로운 토큰을 인식할 수 있도록 하기 위한 과정입니다.
    if tokenizer is not None and tokenizer.pad_token is not None:
        model._____________________ # TODO: token embedding을 추가된 tokenizer를 위해 resize해줍니다.
    
    return model

# 5. 모델 미세 조정(Fine-tuning)
def train_model(model, dataloader, epochs=30, learning_rate=5e-5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    GPT-2 모델을 주어진 데이터로 미세 조정하는 함수
    :param model: 사전 학습된 GPT-2 모델
    :param dataloader: 학습 데이터로더
    :param epochs: 학습 에포크 수 (기본값 30)
    :param learning_rate: 학습률 (기본값 5e-5)
    :param device: 학습에 사용할 장치 (기본값: GPU가 있다면 'cuda', 없으면 'cpu')
    """
    model = model.to(device)  # 모델을 GPU 또는 CPU로 이동
    optimizer = AdamW(model.parameters(), lr=learning_rate)  # AdamW 옵티마이저 설정
    total_steps = len(dataloader) * epochs  # 전체 학습 단계 계산

    # 학습 중 학습률(Learning Rate)을 조절하는 스케줄러 설정
    # 'get_linear_schedule_with_warmup'는 학습률을 선형적으로 감소시키는 스케줄러를 생성하는 함수입니다.
    # 처음에는 'warmup' 단계에서 학습률이 점진적으로 증가하고, 그 이후에는 학습이 진행됨에 따라 학습률이 선형적으로 감소하게 됩니다.
    # 이런 방식을 사용하는 이유는 학습 초반에는 급격한 학습률 변동을 피하고, 안정적으로 학습을 시작한 후 점차 학습률을 낮춰 모델이 수렴하게 하려는 것입니다.
    scheduler = get_linear_schedule_with_warmup(
            optimizer,  # 옵티마이저 (모델의 파라미터 업데이트를 관리)
            
            # 'num_warmup_steps'는 학습 초반에 학습률이 0에서 시작해 점차 증가하는 단계의 스텝 수를 설정합니다.
            # 전체 학습 단계(total_steps)의 10%를 워밍업 스텝으로 설정하는 방식으로, 너무 급격하게 학습률을 높이지 않기 위해 워밍업을 사용합니다.
            num_warmup_steps=int(0.1 * total_steps),  # 워밍업 단계의 스텝 수 (전체 학습 스텝의 10%)
            
            # 'num_training_steps'는 전체 학습 단계의 수를 설정합니다.
            # 학습이 진행되면서 학습률은 'warmup' 후 선형적으로 감소합니다.
            num_training_steps=total_steps  # 전체 학습 스텝 수 설정 (에포크 수 * 배치 수)
        )

    model.train()  # 모델을 학습 모드로 설정
    for epoch in range(epochs):  # 에포크 반복
        print(f"에포크 {epoch + 1}/{epochs}")
        total_loss = 0  # 총 손실 초기화
        for batch in dataloader:  # 미니 배치 반복
            optimizer.zero_grad()  # 기울기 초기화
            
            # TODO: 모델 입력을 위한 데이터
            # 'input_ids': 텍스트 데이터가 토크나이저를 통해 숫자로 변환된 토큰 ID들입니다. 각 단어는 특정 ID로 매핑됩니다.
            # GPT-2 모델은 이 토큰 ID들을 입력으로 받아 학습을 진행합니다. 이 ID들은 모델이 이해할 수 있는 형식으로 변환된 텍스트 데이터입니다.
            input_ids = _____________________.to(device)
            
            # TODO: 'attention_mask'는 패딩된 부분을 무시하기 위한 마스크입니다.
            # 입력 텍스트가 고정된 길이로 맞춰질 때, 짧은 문장은 패딩(예: 0)으로 채워집니다. 
            # 'attention_mask'는 모델이 이러한 패딩된 토큰을 무시하고 실제 데이터를 학습하도록 돕는 역할을 합니다.
            attention_mask = _____________________.to(device)
            
            # TODO: 'labels'는 모델이 예측해야 할 정답입니다.
            # 'labels'는 모델의 출력과 비교할 대상인 실제 정답 토큰 ID입니다.
            # GPT-2 모델은 다음 단어를 예측하는 언어 모델이기 때문에, 학습 시에 이 'labels'를 통해 손실 값을 계산하고 학습을 진행합니다.
            labels = _____________________.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)  # 모델 출력 계산
            loss = outputs.loss  # 손실 계산
            loss.backward()  # 역전파로 기울기 계산

            optimizer.step()  # 옵티마이저 스텝
            scheduler.step()  # 학습률 스케줄러 스텝

            total_loss += loss.item()  # 손실 누적
        avg_loss = total_loss / len(dataloader)  # 평균 손실 계산
        print(f"평균 손실: {avg_loss:.4f}")

    print("모델 미세 조정 완료.")
    return model  # 미세 조정된 모델 반환

# 6. 모델 저장 함수
def save_model(model, tokenizer, save_path='./finetuned_gpt2_traffic'):
    """
    미세 조정된 모델과 토크나이저를 지정된 경로에 저장하는 함수
    :param model: 미세 조정된 GPT-2 모델
    :param tokenizer: GPT-2 토크나이저
    :param save_path: 저장할 경로 (기본값 './finetuned_gpt2_traffic')
    """
    if not os.path.exists(save_path):  # 저장 경로가 없으면 디렉토리 생성
        os.makedirs(save_path)
    model.save_pretrained(save_path)  # 모델 저장
    tokenizer.save_pretrained(save_path)  # 토크나이저 저장

    print(f"모델과 토크나이저가 {save_path}에 저장되었습니다.")

# 7. 저장된 모델과 토크나이저를 불러오는 함수
def load_finetuned_model_and_tokenizer(save_path='./finetuned_gpt2_traffic'):
    """
    저장된 모델과 토크나이저를 불러오는 함수
    :param save_path: 모델과 토크나이저가 저장된 경로 (기본값 './finetuned_gpt2_traffic')
    :return: 불러온 모델과 토크나이저
    """
    if not os.path.exists(save_path):
        raise ValueError(f"{save_path} 경로가 존재하지 않습니다. 먼저 모델을 저장해 주세요.")
    
    # 미세 조정된 모델과 토크나이저로 새롭게 로드
    # TODO: from_pretrained함수로 새로운 모델을 불러와주세요.
    model = GPT2LMHeadModel._____________________
    tokenizer = PreTrainedTokenizerFast._____________________
    
    return model, tokenizer


# 8. 응답 정제 함수
def remove_repeated_sentences(text):
    """
    텍스트에서 반복되는 문장을 제거하는 함수
    :param text: 입력 텍스트
    """
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)  # 문장 분리
    unique_sentences = []  # 고유 문장 리스트
    for sentence in sentences:
        if sentence not in unique_sentences:  # 고유한 문장만 추가
            unique_sentences.append(sentence)
        else:
            break  # 반복되는 문장이 나오면 중단
    return ' '.join(unique_sentences)

def clean_response(response_text, prompt):
    """
    GPT-2 모델의 응답을 정리하는 함수
    :param response_text: 모델 응답 텍스트
    :param prompt: 사용자 입력 프롬프트
    """
    stripped_response = response_text.replace(prompt, '').strip()  # 사용자 입력 제거
    lines = stripped_response.split('\n')  # 라인 분리
    combined_lines = " ".join(line.strip() for line in lines if line.strip())  # 공백 제거 및 라인 병합
    return remove_repeated_sentences(combined_lines)  # 반복되는 문장 제거

# 9. 챗봇 인터페이스
def chat_with_model(model, tokenizer, max_length=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    미세 조정된 GPT-2 모델을 사용하여 사용자와 대화하는 챗봇 함수
    :param model: 미세 조정된 GPT-2 모델
    :param tokenizer: GPT-2 토크나이저
    :param max_length: 생성할 응답의 최대 길이 (기본값 100)
    :param device: 사용할 장치 (기본값: GPU가 있다면 'cuda', 없으면 'cpu')
    """
    model = model.to(device)  # 모델을 GPU 또는 CPU로 이동
    model.eval()  # 모델을 평가 모드로 설정
    print("챗봇: 안녕하세요! 무엇을 도와드릴까요? (종료하려면 'exit' 입력)")

    while True:  # 사용자와 대화 반복
        user_input = input("사용자: ")
        if user_input.lower() in ["exit", "quit"]:  # 대화 종료 조건
            print("챗봇: 대화를 종료합니다.")
            break

        # 챗봇의 입력은 String으로 받습니다. 해당 인풋을 model이 인식할 수 있는 token 형태로 변환해줘야 합니다.
        # TODO: encode함수를 통해서 해당 input_ids를 만들어주세요.
        input_ids = _____________________.to(device)  # 사용자 입력 토큰화 및 장치 이동
        
        with torch.no_grad():  # 기울기 계산 없이 모델 생성
            output = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)  # 모델 응답 디코딩
        response = clean_response(response, user_input)  # 응답 정리
        print(f"챗봇: {response}")

# 10. 메인 함수
def main():
    """
    데이터셋 로드, 모델 미세 조정, 챗봇 실행을 위한 메인 함수
    """
    file_path = 'traffic_data.csv'
    if not os.path.exists(file_path):  # 데이터 파일 존재 여부 확인
        print(f"{file_path} 파일이 존재하지 않습니다. 데이터를 먼저 준비해주세요.")
        return
    
    dataset = load_dataset_and_preprocess(file_path=file_path, model_name='skt/kogpt2-base-v2', max_length=128)  # 데이터 로드 및 전처리
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=True)  # 데이터로더 생성
    
    model = load_model(model_name='skt/kogpt2-base-v2', tokenizer=load_tokenizer(model_name='skt/kogpt2-base-v2'))  # 모델 로드
    
    # 모델 미세 조정
    model = train_model(model, dataloader, epochs=3, learning_rate=5e-5)  # 모델 미세 조정

    # 미세 조정된 모델과 토크나이저 불러옴
    model, tokenizer = load_finetuned_model_and_tokenizer(save_path='./finetuned_gpt2_traffic')

    # 챗봇 실행
    # TODO: chatbot을 실행하는 함수를 실행해주면 됩니다.
    _____________________  # 챗봇 시작

# 메인 함수 실행
if __name__ == "__main__":
    main()
