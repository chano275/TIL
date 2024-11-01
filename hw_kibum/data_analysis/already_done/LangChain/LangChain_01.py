import os, time

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage

from langchain_community.document_loaders import TextLoader, PyPDFLoader, YoutubeLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS

from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings, OpenAI

from langchain_core.prompts import ChatPromptTemplate

import bs4


"""
pip install langchain langchain-community langchain-openai
pip install youtube-transcript-api pytube  # 지금은 작동 안하는듯? 유튜브 영상 자막 받아와 txt로 하는 ?

LangChain 및 관련 모듈 불러오기
> 텍스트 분할, 벡터 저장소(FAISS), OpenAI 기반 임베딩, OpenAI 대화 모델을 사용하기 위한 모듈을 불러옵니다.

PyPDFLoader: PDF 파일을 로드하여 텍스트 데이터를 추출하는 모듈
YoutubeLoader: 유튜브 자막을 불러와서 문서로 로드하는 모듈
WebBaseLoader: 웹 페이지의 데이터를 불러오는 모듈

ChatOpenAI   : OpenAI의 GPT 모델을 사용하여 대화를 처리하는 모듈
HumanMessage : 사용자가 입력한 질문이나 메시지를 GPT 모델에 전달하기 위해 사용
SystemMessage: 시스템의 역할을 정의하고 모델에게 초기 지침을 제공하기 위해 사용
CharacterTextSplitter: 문서를 여러 청크로 나누기 위한 텍스트 분할기
FAISS : 텍스트 데이터 [ 청크 ] 를 벡터로 변환하여 검색할 수 있도록 지원하는 라이브러리
      : 대규모 벡터 데이터를 저장하고 유사도를 기반으로 검색할 수 있는 라이브러리

OpenAIEmbeddings: [ 텍스트 > 벡터 ] 로 변환하는 OpenAI 임베딩 모델


TextLoader() 참고: https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/
loader.load() 참고: https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/
FAISS.from_documents() 함수 참고: https://sj-langchain.readthedocs.io/en/latest/vectorstores/langchain.vectorstores.faiss.FAISS.html#langchain.vectorstores.faiss.FAISS.from_documents

OpenAI API 키 설정 > TODO: os.environ을 사용하여 "OPENAI_API_KEY"를 설정

____________________________________________________________
[ 환경 변수(os.environ) 참고: https://docs.python.org/3/library/os.html#os.environ ]
[ cf > 코드에 직접 포함하는 방식 vs os.environ을 사용하는 방식 ]
1. 코드에 직접 포함하는 방식
장점: 코드를 실행할 때마다 API 키를 설정할 필요가 없습니다. 그래서 테스트 환경에서 사용하는 것이 편리합니다.
단점: 코드에 API 키가 포함되어 있기 때문에 보안상 취약합니다.

2. os.environ을 사용하는 방식
장점: API 키를 환경 변수로 설정하므로 코드에 API 키가 포함되지 않아 보안상 안전합니다. 그래서 실제 프로덕션 환경에서 사용하는 것이 좋습니다.
단점: 코드를 실행할 때마다 API 키를 설정해야 하므로 번거롭습니다.
____________________________________________________________
"""
os.environ["OPENAI_API_KEY"] = ''

# TODO: txt 파일 🔥
# loader = TextLoader('./data/financial_articles.txt')  # Langchain의 TextLoader를 사용하여 금융 기사 txt 파일을 로드
# documents = loader.load()                             # loader.load() 메서드를 사용하여 문서 내용을 로드 => 로드된 문서를 documents 변수에 저장
#
# for i, doc in enumerate(documents):                   # 각 문서의 텍스트 길이와 토큰 수를 출력
#     print(f"문서 {i+1}의 길이: {len(doc.page_content)} 문자")
#     print(f"문서 {i+1}의 토큰 수: {len(doc.page_content.split())} 토큰")
#
# print("\n첫 번째 문서의 내용:")
# print(documents[0].page_content)                      # 첫 번째 문서의 내용을 출력
# # _________________________________________________________________________________
# loader = TextLoader("./data/financial_articles.txt")       # 1. 금융 기사 데이터를 로드
# documents = loader.load()
#
# embedding = OpenAIEmbeddings()                             # 2. OpenAI 임베딩과 벡터스토어 생성
# vector_store = FAISS.from_documents(documents, embedding)  # FAISS.from_documents() 메서드를 사용하여 vector_store 변수를 초기화
#                                                            # documents와 embedding을 사용해 벡터스토어를 생성합니다.
#
# llm = OpenAI()                                             # 3. RAG 알고리즘을 사용한 질문 응답 시스템 구성
#
# # 프롬프트 설정
# system_prompt = ("주어진 컨텍스트를 사용하여 질문에 답변하세요. " "모르면 '모릅니다'라고 말하세요. " "최대 세 문장으로 간결하게 답변하세요. " "컨텍스트: {context}")
# prompt = ChatPromptTemplate.from_messages( [("system", system_prompt), ("human", "{input}"),] )
#
# # 질문 응답 체인 생성
# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# qa_chain = create_retrieval_chain(vector_store.as_retriever(), question_answer_chain)
#
# # 4. 질문 리스트에 대한 답변 생성
# questions = ["최근 금융 시장의 주요 동향은 무엇인가요?", "금리 인상의 영향은 어떻게 분석될 수 있나요?", "현재 암호화폐 시장의 상황은 어떤가요?"]
#
# for question in questions:  # 각 질문에 대해 답변 생성 및 출력
#     print('#################################################################')
#     answer = qa_chain.invoke({"input": question})
#
#     # 결과에서 'answer' 키를 찾고 출력, 없으면 기본값으로 '답변을 찾을 수 없습니다.' 설정
#     response = answer.get('answer', '답변을 찾을 수 없습니다.')
#     print(f"질문: {question}")
#     print(f"답변: {response}\n")



# TODO: LLM 이용 🔥

# # 질문을 처리하는 함수 정의
# def handle_traffic_query(query_text):
#     llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0) # OpenAI의 GPT-4 모델을 불러옵니다. 여기서는 작은 버전(gpt-4o-mini)을 사용합니다.
#     system_message = SystemMessage(content="너는 교통 정보 전문가야. 사용자의 질문에 대해 교통 관련 정보를 바탕으로 답변해줘.")  # 모델에게 교통 전문가로서 응답하도록 시스템 메시지 설정
#     human_message = HumanMessage(content=f"교통 질문: {query_text}")  # 사용자의 교통 관련 질문을 전달하는 메시지 생성
#     conversation = [system_message, human_message]    # 대화에 시스템 메시지와 사용자 질문을 추가
#     response = llm.__call__(conversation)             # 모델에게 대화 형식으로 질문을 보내고 답변을 받습니다.
#     return response.content                           # 결과 반환 (모델의 답변)
#
# # 질문 처리
# traffic_query = "서울에서 부산까지 고속도로 교통 상황은 어떤가요?"  # 예시 질문 설정
# response = handle_traffic_query(traffic_query)  # 질문에 대한 답변을 처리하는 함수 호출
# print(f"답변:\n {response}")  # 결과 출력


# TODO: [웹 / PDF] LOAD 🔥

# # 아래의 load_traffic_data만 교체해주면 똑같은 동작 함
# # 웹에서 데이터를 불러오는 함수 - 주어진 웹 페이지 URL에서 데이터를 불러옵니다.
# def load_traffic_news(url):
#     loader = WebBaseLoader(url)                                 # 웹 로더 생성
#     documents = loader.load()                                   # 웹 페이지에서 문서 불러오기
#     print(f"{len(documents)}개의 페이지에서 데이터를 로드했습니다.")  # 로드된 문서 수 출력
#     return documents                                            # 로드된 문서 반환

# # PDF 파일 로드 함수 - PDF 파일을 로드해 텍스트 추출 > 각 페이지가 하나의 문서로 반환
# def load_traffic_data(pdf_file_path):
#     loader = PyPDFLoader(pdf_file_path)                         # PDF 로더 생성
#     documents = loader.load()                                   # PDF에서 문서(페이지) 불러오기
#     print(f"{len(documents)}개의 페이지에서 텍스트를 로드했습니다.")  # 페이지 수 출력
#     return documents                                            # 로드된 문서 반환
#

# # 문서를 일정 크기의 청크로 나누어 모델이 처리할 수 있는 단위로 나누는 함수
# def split(documents):
#     # 청크를 분할할 때 사용할 구분자 (줄바꿈 기준) / 한 청크의 최대 길이 (단위: 문자 수) /  청크 간 겹치는 부분 길이 (중복된 부분을 포함하여 문맥 유지)
#     text_splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=100)
#     splits = text_splitter.split_documents(documents)  # 문서를 청크로 나누기
#     print(f"{len(splits)}개의 청크로 나누었습니다.")       # 청크 수 출력
#     return splits                                       # 나눈 청크 반환
#
# # 청크를 벡터로 변환하여 벡터 스토어에 저장하는 함수 - OpenAI 임베딩을 사용해 청크를 벡터로 변환하고, FAISS 벡터 스토어에 저장
# def store_in_vector_db(splits):
#     embeddings = OpenAIEmbeddings()                          # OpenAI 임베딩 모델 사용
#     vector_store = FAISS.from_documents(splits, embeddings)  # 청크를 벡터로 변환하고 저장
#     print("벡터 스토어에 청크를 저장했습니다.")                   # 저장 완료 메시지 출력
#     return vector_store                                      # 벡터 스토어 반환
#
# # 입력한 질문(query)과 가장 유사한 문서를 검색하는 함수
# def retrieve_similar_docs(query_text, vector_store):
#     docs = vector_store.similarity_search(query_text, k=3)  # 유사도 높은 상위 3개의 문서 검색
#     print("유사한 문서를 검색했습니다.")                        # 검색 완료 메시지 출력
#     return docs                                             # 검색된 문서 반환
#
# # 검색된 문서를 바탕으로 LLM을 사용하여 질문에 대한 답변을 생성하는 함수
# def generate_answer(query_text, docs):
#     llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)   # OpenAI의 GPT-4 모델을 불러옵니다.
#     system_message = SystemMessage(content="너는 교통 전문가야. 질문에 대해 관련된 교통 데이터를 바탕으로 답변해줘.")  # 모델의 역할을 지정 (교통 전문가)
#     human_message = HumanMessage(content=f"질문: {query_text}\n\n{docs}")  # 질문과 검색된 문서로 유저 메시지 생성
#     conversation = [system_message, human_message]             # 대화에 시스템 메시지와 유저 질문 추가
#     response = llm.__call__(conversation)                      # 모델에게 대화 전달하여 답변 생성
#     return response.content                                    # 생성된 답변 반환
#
# ## Main - 택 1

# pdf_file_path = "./data/교통_3대_혁신_전략.pdf"  # PDF 파일 경로 설정
# query_text = "GTX의 A노선의 구간은?"              # 검색할 질문 설정
# documents = load_traffic_data(pdf_file_path)                    # 1. Loading: PDF 파일에서 문서 로드
#
# url = "https://www.ohmynews.com/NWS_Web/View/at_pg.aspx?CNTN_CD=A0003069986&CMPT_CD=P0010&utm_source=naver&utm_medium=newsearch&utm_campaign=naver_news"  # 웹 페이지 URL 설정
# documents = load_traffic_news(url)
# query_text = "차량신호가 녹색인 경우 우회전은 어떻게 해야해?"  # 검색할 질문 설정
#
#
# splits = split(documents)                                       # 2. Splitting: 문서를 여러 청크로 분할
# vector_store = store_in_vector_db(splits)                       # 3. Storage : 청크를 벡터로 변환 후 벡터 스토어에 저장
# similar_docs = retrieve_similar_docs(query_text, vector_store)  # 4. Retrieval: 질문에 대한 유사한 문서 검색
# answer = generate_answer(query_text, similar_docs)              # 5. Generation: 검색된 문서를 바탕으로 답변 생성
# print(f"최종 답변: {answer}")                                    # 최종 결과 출력 (모델이 생성한 답변 출력)




# TODO: [유튜브 자막] LOAD 🔥

# # 유튜브 자막을 불러오는 함수 - 유튜브 URL을 입력받아 자막 데이터를 로드하고, 성공할 때까지 재시도
# def load_youtube_transcript(url, add_video_info=True, max_retries=5, retry_delay=2):
#     attempts = 0  # 시도 횟수 초기화
#     while attempts < max_retries:  # 최대 시도 횟수까지 반복
#         try:
#             # 유튜브 자막 로더를 URL을 기반으로 생성 -                   비디오 정보 포함 여부 / 한국어와 영어 자막 지원
#             loader = YoutubeLoader.from_youtube_url(url, add_video_info=add_video_info, language=['ko', 'en'],)
#             documents = loader.load()                           # 유튜브 자막 로드
#             print(f"유튜브 자막을 성공적으로 불러왔습니다.")
#             return documents                                    # 성공 시 로드된 문서 반환
#         except Exception as e:                                  # 오류 발생 시
#             attempts += 1                                       # 시도 횟수 증가
#             print(f"유튜브 자막을 불러오는 중 오류 발생: {e}. {retry_delay}초 후 재시도합니다. (시도 {attempts}/{max_retries})")
#             time.sleep(retry_delay)                             # 재시도 전 대기
#     print("최대 시도 횟수를 초과했습니다. 자막을 불러오지 못했습니다.")
#     return None                                                 # 실패 시 None 반환
#
#
# # LLM을 사용하여 답변 생성 - 불러온 유튜브 자막을 바탕으로 질문에 답변을 생성하는 함수
# def generate_answer(query_text, docs):
#     llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)    # OpenAI의 GPT-4 모델 불러오기
#     system_message = SystemMessage(content="너는 유튜브 자막 내용을 바탕으로 질문에 답변하는 역할을 한다.")  # 시스템 메시지를 통해 모델의 역할을 지정 (자막을 바탕으로 답변하는 역할)
#     human_message = HumanMessage(content=f"질문: {query_text}\n\n{docs}")    # 유저의 질문을 담은 메시지 생성
#     conversation = [system_message, human_message]    # 대화에 시스템 메시지와 유저 질문 추가
#     response = llm.__call__(conversation)    # 모델에게 대화를 전달하여 답변 생성
#     return response.content  # 생성된 답변 반환
#
# # Main : 유튜브 자막이 있는 교통 뉴스 영상 URL과 질문을 설정
# url = "https://www.youtube.com/watch?v=_8w803FPWmw"  # 유튜브 영상 URL (교통 뉴스)
# query_text = "이 뉴스의 주요 내용은 무엇인가요?"         # 유저가 묻는 질문
# documents = load_youtube_transcript(url)         # 1. Loading: 유튜브 자막을 로드
# answer = generate_answer(query_text, documents)  # 2. Generation: 자막을 바탕으로 질문에 대한 답변 생성
# print(f"최종 답변: {answer}")                     # 모델이 생성한 최종 답변 출력
