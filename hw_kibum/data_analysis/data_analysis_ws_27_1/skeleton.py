# Step 1: 필요한 라이브러리 설치
# !pip install langchain langchain-community langchain-openai

# Step 2: LangChain 및 관련 모듈 불러오기
# 텍스트 분할, 벡터 저장소(FAISS), OpenAI 기반 임베딩, OpenAI 대화 모델을 사용하기 위한 모듈을 불러옵니다.

# 주요 모듈에 대한 설명
# ChatOpenAI: OpenAI의 GPT 모델을 사용하여 대화를 처리하는 모듈입니다.
# HumanMessage: 사용자가 입력한 질문이나 메시지를 GPT 모델에 전달하기 위해 사용됩니다.
# SystemMessage: 시스템의 역할을 정의하고 모델에게 초기 지침을 제공하기 위해 사용됩니다.
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


import os

# OpenAI API 키 설정 (실제 API 키로 변경 필요)
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Step 3: 교통 관련 질문을 처리하는 함수 정의
# TODO: 위에서 불러온 모듈을 기반으로 빈칸을 완성하세요.
def handle_traffic_query(query_text):
    # OpenAI의 GPT-4 모델을 불러옵니다. 여기서는 작은 버전(gpt-4o-mini)을 사용합니다.
    llm = _______________(model_name="gpt-4o-mini", temperature=0)

    # 모델에게 교통 전문가로서 응답하도록 시스템 메시지 설정
    system_message = _______________(content="너는 교통 정보 전문가야. 사용자의 질문에 대해 교통 관련 정보를 바탕으로 답변해줘.")
    
    # 사용자의 교통 관련 질문을 전달하는 메시지 생성
    human_message = _______________(content=f"교통 질문: {query_text}")

    # 대화에 시스템 메시지와 사용자 질문을 추가
    conversation = [system_message, human_message]

    # 모델에게 대화 형식으로 질문을 보내고 답변을 받습니다.
    response = llm._______________(conversation)

    # 결과 반환 (모델의 답변)
    return response.content

# Step 4: 교통 관련 예시 질문 처리
# 예시 질문 설정
traffic_query = "서울에서 부산까지 고속도로 교통 상황은 어떤가요?"

# 질문에 대한 답변을 처리하는 함수 호출
response = handle_traffic_query(traffic_query)

# 결과 출력
print(f"답변:\n {response}") 
