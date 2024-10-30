# pip install youtube-transcript-api pytube

# 필요한 모듈 불러오기
# FAISS: 대규모 벡터 데이터를 저장하고 유사도를 기반으로 검색할 수 있는 라이브러리
# YoutubeLoader: 유튜브 자막을 불러와서 문서로 로드하는 모듈
# CharacterTextSplitter: 문서를 여러 청크로 나누기 위한 텍스트 분할기
# OpenAIEmbeddings: 텍스트 데이터를 벡터로 변환하는 OpenAI 임베딩 모델
# ChatOpenAI: OpenAI의 GPT 모델을 사용하여 대화를 처리하는 모듈
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import time
import os

# TODO: 위에서 불러온 모듈을 기반으로 빈칸을 완성하세요.

# OpenAI API 키 입력 (실제 API 키로 변경)
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Step 1: 유튜브 자막을 불러오는 함수
# 유튜브 URL을 입력받아 자막 데이터를 로드하고, 성공할 때까지 재시도합니다.
def load_youtube_transcript(url, add_video_info=True, max_retries=5, retry_delay=2):
    attempts = 0  # 시도 횟수 초기화
    while attempts < max_retries:  # 최대 시도 횟수까지 반복
        try:
            # 유튜브 자막 로더를 URL을 기반으로 생성
            loader = ______________.from_youtube_url(
                url,
                add_video_info=add_video_info,  # 비디오 정보 포함 여부
                language=['ko', 'en'],  # 한국어와 영어 자막 지원
            )
            documents = loader.load()  # 유튜브 자막 로드
            print(f"유튜브 자막을 성공적으로 불러왔습니다.")
            return documents  # 성공 시 로드된 문서 반환
        except Exception as e:  # 오류 발생 시
            attempts += 1  # 시도 횟수 증가
            print(f"유튜브 자막을 불러오는 중 오류 발생: {e}. {retry_delay}초 후 재시도합니다. (시도 {attempts}/{max_retries})")
            time.sleep(retry_delay)  # 재시도 전 대기
    print("최대 시도 횟수를 초과했습니다. 자막을 불러오지 못했습니다.")
    return None  # 실패 시 None 반환

# Step 2: LLM을 사용하여 답변 생성
# 불러온 유튜브 자막을 바탕으로 질문에 답변을 생성하는 함수
def generate_answer(query_text, docs):
    # OpenAI의 GPT-4 모델 불러오기
    llm = ______________(model_name="gpt-4o-mini", temperature=0)
    
    # 시스템 메시지를 통해 모델의 역할을 지정 (자막을 바탕으로 답변하는 역할)
    system_message = ______________(content="너는 유튜브 자막 내용을 바탕으로 질문에 답변하는 역할을 한다.")
    
    # 유저의 질문을 담은 메시지 생성
    human_message = ______________(content=f"질문: {query_text}\n\n{docs}")
    
    # 대화에 시스템 메시지와 유저 질문 추가
    conversation = [system_message, human_message]
    
    # 모델에게 대화를 전달하여 답변 생성
    response = llm.______________(conversation)
    
    return response.content  # 생성된 답변 반환

# Step 3: 통합된 예시 사용
# 유튜브 자막이 있는 교통 뉴스 영상 URL과 질문을 설정
url = "https://www.youtube.com/watch?v=_8w803FPWmw"  # 유튜브 영상 URL (교통 뉴스)
query_text = "이 뉴스의 주요 내용은 무엇인가요?"  # 유저가 묻는 질문

# 1. Loading: 유튜브 자막을 로드
documents = load_youtube_transcript(url)  # 유튜브 자막 로드

# 2. Generation: 자막을 바탕으로 답변 생성
answer = generate_answer(query_text, documents)  # 자막 기반으로 질문에 대한 답변 생성

# 최종 답변 출력
print(f"최종 답변: {answer}")  # 모델이 생성한 답변 출력