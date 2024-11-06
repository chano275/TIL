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

# Step 1: 교통데이터 증강을 위한 GenAI 준비
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

print(f"gold_samples: {gold_samples}")