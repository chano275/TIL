# TODO: 해당 패키지를 설치해야 합니다.
# 필요 패키지 설치
# pip install -qU langchain-ollama==0.1.3 langchain-anthropic langchain-openai scikit-learn ollama

# ollama 설치 및 실행
# curl -fsSL https://ollama.com/install.sh | sh
# ollama serve
# ollama pull llama3.2:1b or llama3.2:3b

from langchain_core.prompts import PromptTemplate  # 프롬프트 템플릿 생성용
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # OpenAI 모델 및 임베딩 기능
from langchain_ollama import ChatOllama # Ollama 기반 모델(local llama를 위한 모델) 
from langchain_anthropic import ChatAnthropic # claude 모델
from sklearn.metrics.pairwise import cosine_similarity  # 유사도 계산용
import numpy as np  # 배열 연산을 위한 패키지
import os  # 시스템 환경 변수 및 파일 경로 관련
import random  # 랜덤 샘플링을 위한 모듈

# 교통데이터 증강을 위한 GenAI 준비
# 환경 변수와 모델 설정
# TODO: 각 모델(GPT, Anthropic)의 API key를 발급받습니다.
os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""


# 임베딩 모델 설정 (공통)
# OpenAI 임베딩 모델로 텍스트를 수치화하여 유사도 계산 가능하게 준비
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 시드 키워드 생성
# 교통과 관련된 주요 키워드를 기반으로 데이터 생성 방향을 결정
traffic_keywords = ["신호등 고장", "차량 정체", "교통사고"]

# 골드 샘플 생성
# 모델이 학습할 수 있는 표본 답변 예시 (Few-shot 학습에 사용)
# gold_samples 데이터를 상황별로 분류하여 딕셔너리로 정리
gold_samples = [
    # 신호등 고장 관련 모범 답변 10개
    {"situation": "신호등 고장", "response": "신호등이 고장났을 때는 교차로에서 일시 정지 후 좌우를 살피고 진행하세요."},
    {"situation": "신호등 고장", "response": "신호등 고장 시 교통 경찰의 수신호를 따라주세요."},
    {"situation": "신호등 고장", "response": "교차로에서 신호등이 작동하지 않으면 다른 차량에게 양보 운전을 실천하세요."},
    {"situation": "신호등 고장", "response": "신호등 고장 상황에서는 서행 운전으로 안전에 유의하세요."},
    {"situation": "신호등 고장", "response": "신호등이 깜빡일 경우 주의 표지로 간주하고 운전하세요."},
    {"situation": "신호등 고장", "response": "신호등 고장을 발견하면 관할 당국에 신고해주세요."},
    {"situation": "신호등 고장", "response": "신호등이 작동하지 않는 교차로에서는 항상 일시 정지하세요."},
    {"situation": "신호등 고장", "response": "보행자 신호등이 고장났을 때는 보행자의 통행에 주의하세요."},
    {"situation": "신호등 고장", "response": "야간에 신호등 고장 시 전조등을 활용하여 의사소통하세요."},
    {"situation": "신호등 고장", "response": "신호등 고장으로 혼잡한 교차로에서는 우회 경로를 고려하세요."},
    # TODO: few shot 학습을 위해 차량 정체 관련 모범 답변 10개와 교통사고 관련 모범 답변 10개를 GPT를 통해 만들어 위와 같이 만들어서 넣어주세요.
    # 차량 정체 관련 모범 답변 10개

    # 교통사고 관련 모범 답변 10개
]

# 데이터 생성을 위한 공통 함수 정의
def generate_data(model_name, traffic_keywords, gold_samples, max_data=15, k=5):
    """
    데이터 증강을 위해 선택된 모델을 사용하여 새로운 데이터를 생성.
    
    Parameters:
    - model_name (str): 사용할 모델 이름 ('gpt', 'claude', 'llama' 중 하나)
    - traffic_keywords (list): 각 모델에 입력할 교통 관련 키워드
    - gold_samples (list): 모델에 학습시킬 예시 답변 리스트
    - max_data (int): 생성할 데이터 최대 수
    - k (int): 한 번에 생성할 답변 수

    Returns:
    - unique_texts (list): 최종적으로 필터링된 고유한 텍스트들
    - unique_embeddings (list): 각 텍스트의 임베딩 리스트
    """
    total_generated = 0  # 총 생성된 데이터 수 카운트
    unique_embeddings = []  # 고유 임베딩 저장
    unique_texts = []  # 고유 텍스트 저장

    # 모델 선택
    # 모델 선택을 통해 특정 모델의 응답 스타일을 반영하여 데이터 증강
    # TODO: model_name이 gpt인 경우 ChatOpenAI 모델을 생성해서 데이터 증강에 사용할 gpt-4o-mini 모델을 불러와주세요.
    # TODO: model_name이 claude인 경우ChatAnthropic로 모델을 생성해서 데이터 증강에 사용할 claude-3-5-sonnet-latest 모델을 불러와주세요.
    if model_name == 'gpt':
        model = _____________________  # ChatOpenAI 라이브러리를 이용한 GPT 모델 사용
    elif model_name == 'claude':
        model = _____________________  # ChatAnthropic 라이브러리 이용
    else:
        raise ValueError("지원되지 않는 모델입니다.")  # 오류 처리

    # 각 키워드에 대해 데이터 생성 시작
    for keyword in traffic_keywords:
        if total_generated >= max_data:
            break  # 최대 데이터 수 도달 시 루프 종료

        # 골드 샘플에서 3개의 예시를 랜덤 선택하여 Few-shot 학습에 사용
        few_shot_examples = random.sample([sample['response'] for sample in gold_samples if sample["situation"] == keyword], 3)
        few_shot_text = "\n".join(few_shot_examples)  # 선택된 예시를 문자열로 변환

        # 프롬프트 생성: 모델에게 생성할 내용을 지시
        # TODO: 교통 상황에 대한 모범 답변 few_shot_text를 기반으로 keyword에 맞게 예시를 k가지를 작성할 수 있도록 프롬프트를 작성하세요.
        prompt = f"""

        """

        # 모델을 사용하여 응답 생성: k개 이상을 생성할 때 까지 반복합니다.
        # TODO: Chat model이 prompt를 적용해 응답 결과를 받을 수 있도록 하세요. (이전에 langchain을 이용하면서 무수히 많이 사용하였습니다.)
        generated_texts = []
        while len(generated_texts) != k:
            response = model.______________(prompt)
            generated = response.content.strip().split('\n')  # 응답을 텍스트로 분할하여 리스트로 저장

            # 응답에서 빈 문자열 제거 및 공백 제거
            generated = [text.strip() for text in generated if text.strip() != ''][:k]
            generated_texts += generated
        
            generated_texts = generated_texts[:k]

        # 생성된 각 텍스트에 대해 유사도 계산 및 필터링
        for text in generated_texts:
            if total_generated >= max_data:
                break

            # 텍스트 임베딩 생성
            emb = embedding_model.embed_query(text)  # 생성된 텍스트를 임베딩화

            # 고유한 임베딩과 텍스트를 저장
            # TODO: 각각에 저장되어야할 값을 넣어주세요. embedding값과 text 자체를 가진 리스트를 각각 선언합니다.
            unique_embeddings.append(___________)
            unique_texts.append(___________)
            total_generated += 1
            print(f"{model_name.upper()} - Generated text for '{keyword}': {text}")

    return unique_texts, unique_embeddings

# Claude를 사용한 교통 데이터 증강
# Claude 모델로 증강된 텍스트 및 임베딩 생성
# TODO: generate_data 함수를 통해 'claude' 모델, 키워드 리스트, 예시 답변 리스트, 최대 생성 데이터 수=15, 한 번에 생성할 답변 수= 5로 선언하세요. 
claude_texts, claude_embeddings = ________________________________
