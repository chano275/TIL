# 필요한 모듈 불러오기
# FAISS: 대규모 벡터 데이터를 저장하고 유사도를 기반으로 검색할 수 있는 라이브러리
# WebBaseLoader: 웹 페이지의 데이터를 불러오는 모듈
# CharacterTextSplitter: 문서를 여러 청크로 나누기 위한 텍스트 분할기
# OpenAIEmbeddings: 텍스트 데이터를 벡터로 변환하는 OpenAI 임베딩 모델
# ChatOpenAI: OpenAI의 GPT 모델을 사용하여 대화를 처리하는 모듈
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

import os

# TODO: 위에서 불러온 모듈을 기반으로 빈칸을 완성하세요.

# OpenAI API 키 설정 (실제 API 키로 변경 필요)
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Step 1: 웹에서 데이터를 불러오는 함수
# 주어진 웹 페이지 URL에서 데이터를 불러옵니다.
def load_traffic_news(url):
    loader = ______________(url)  # 웹 로더 생성
    documents = loader.load()  # 웹 페이지에서 문서 불러오기
    print(f"{len(documents)}개의 페이지에서 데이터를 로드했습니다.")  # 로드된 문서 수 출력
    return documents  # 로드된 문서 반환

# Step 2: 문서를 청크로 나누는 함수
# 문서들을 일정 크기의 청크로 나누어 모델이 처리할 수 있는 단위로 나눕니다.
def split_documents(documents):
    text_splitter = ______________(
        separator="\n",  # 청크를 분할할 때 사용할 구분자 (줄바꿈 기준)
        chunk_size=300,  # 한 청크의 최대 길이 (단위: 문자 수)
        chunk_overlap=100  # 청크 간 겹치는 부분 길이 (중복된 부분을 포함하여 문맥 유지)
    )
    splits = text_splitter.______________(documents)  # 문서를 청크로 나누기
    print(f"{len(splits)}개의 청크로 나누었습니다.")  # 청크 수 출력
    return splits  # 나눈 청크 반환

# Step 3: 청크를 벡터로 변환하여 벡터 스토어에 저장하는 함수
# OpenAI 임베딩을 사용해 청크를 벡터로 변환하고, FAISS 벡터 스토어에 저장합니다.
def store_in_vector_db(splits):
    embeddings = ______________()  # OpenAI 임베딩 모델 사용
    vector_store = ______________.from_documents(splits, embeddings)  # 청크를 벡터로 변환하고 저장
    print("벡터 스토어에 청크를 저장했습니다.")  # 저장 완료 메시지 출력
    return vector_store  # 벡터 스토어 반환

# Step 4: 유사한 문서를 검색하는 함수
# 입력한 질문(query)과 가장 유사한 문서를 검색합니다.
def retrieve_similar_docs(query_text, vector_store):
    docs = vector_store.______________(query_text, k=3)  # 유사도 높은 상위 3개의 문서 검색
    print("유사한 문서를 검색했습니다.")  # 검색 완료 메시지 출력
    return docs  # 검색된 문서 반환

# Step 5: LLM을 사용하여 답변 생성하는 함수
# 검색된 문서를 바탕으로 모델이 답변을 생성합니다.
def generate_answer(query_text, docs):
    # OpenAI의 GPT-4 모델을 불러옵니다.
    llm = ______________(model_name="gpt-4o-mini", temperature=0)
    
    # 시스템 메시지를 통해 모델의 역할을 지정 (교통 전문가로서 답변)
    system_message = ______________(content="너는 교통 전문가야. 질문에 대해 관련된 교통 데이터를 바탕으로 답변해줘.")
    
    # 유저의 질문과 검색된 문서를 담은 메시지 생성
    human_message = ______________(content=f"질문: {query_text}\n\n{docs}")
    
    # 대화에 시스템 메시지와 유저 질문 추가
    conversation = [system_message, human_message]
    
    # 모델에게 대화를 전달하여 답변 생성
    response = llm.______________(conversation)
    
    return response.content  # 생성된 답변 반환

# Step 6: 예시 사용
# 웹 페이지 URL과 질문을 설정
url = "https://www.ohmynews.com/NWS_Web/View/at_pg.aspx?CNTN_CD=A0003069986&CMPT_CD=P0010&utm_source=naver&utm_medium=newsearch&utm_campaign=naver_news"  # 웹 페이지 URL 설정
query_text = "차량신호가 녹색인 경우 우회전은 어떻게 해야해?"  # 검색할 질문 설정

# 1. Loading: 웹에서 데이터를 로드
documents = load_traffic_news(url)  # 웹 페이지에서 문서 로드

# 2. Splitting: 문서를 청크로 나누기
splits = split_documents(documents)  # 문서를 여러 청크로 분할

# 3. Storage: 벡터 스토어에 저장
vector_store = store_in_vector_db(splits)  # 청크를 벡터로 변환 후 벡터 스토어에 저장

# 4. Retrieval: 질문에 대한 유사한 문서 검색
similar_docs = retrieve_similar_docs(query_text, vector_store)  # 유사한 문서 검색

# 5. Generation: 검색된 문서를 바탕으로 답변 생성
answer = generate_answer(query_text, similar_docs)  # 질문에 대한 답변 생성

# 최종 답변 출력
print(f"최종 답변: {answer}")  # 모델이 생성한 답변 출력
