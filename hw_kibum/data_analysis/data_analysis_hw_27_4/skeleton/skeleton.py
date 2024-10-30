import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# OpenAI API 키 설정
# TODO: os.environ을 사용하여 "OPENAI_API_KEY"를 설정하세요.
# os.environ을 사용해 "OPENAI_API_KEY"를 설정합니다.
# 코드에 직접 포함하는 방식 vs os.environ을 사용하는 방식
# 1. 코드에 직접 포함하는 방식
# 장점: 코드를 실행할 때마다 API 키를 설정할 필요가 없습니다. 그래서 테스트 환경에서 사용하는 것이 편리합니다.
# 단점: 코드에 API 키가 포함되어 있기 때문에 보안상 취약합니다.
# 2. os.environ을 사용하는 방식
# 장점: API 키를 환경 변수로 설정하므로 코드에 API 키가 포함되지 않아 보안상 안전합니다. 그래서 실제 프로덕션 환경에서 사용하는 것이 좋습니다.
# 단점: 코드를 실행할 때마다 API 키를 설정해야 하므로 번거롭습니다.
# 환경 변수(os.environ) 참고: https://docs.python.org/3/library/os.html#os.environ
os.environ["OPENAI_API_KEY"] = ''
# 1. 금융 기사 데이터를 로드
loader = TextLoader("../data/financial_articles.txt")  # 금융 기사 파일 경로
documents = loader.load()

# 2. OpenAI 임베딩과 벡터스토어 생성
embedding = OpenAIEmbeddings()

# TODO: FAISS.from_documents() 메서드를 사용하여 vector_store 변수를 초기화하세요.
# FAISS.from_documents() 메서드를 사용하여 vector_store 변수를 초기화합니다.
# documents와 embedding을 사용해 벡터스토어를 생성합니다.
# FAISS.from_documents() 함수 참고: https://sj-langchain.readthedocs.io/en/latest/vectorstores/langchain.vectorstores.faiss.FAISS.html#langchain.vectorstores.faiss.FAISS.from_documents
vector_store = FAISS.from_documents(documents, embedding)

# 3. RAG 알고리즘을 사용한 질문 응답 시스템 구성
llm = OpenAI()

# 프롬프트 설정
system_prompt = (
    "주어진 컨텍스트를 사용하여 질문에 답변하세요. "
    "모르면 '모릅니다'라고 말하세요. "
    "최대 세 문장으로 간결하게 답변하세요. "
    "컨텍스트: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 질문 응답 체인 생성
question_answer_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(vector_store.as_retriever(), question_answer_chain)

# 4. 질문 리스트에 대한 답변 생성
questions = [
    "최근 금융 시장의 주요 동향은 무엇인가요?",
    "금리 인상의 영향은 어떻게 분석될 수 있나요?",
    "현재 암호화폐 시장의 상황은 어떤가요?"
]

# 각 질문에 대해 답변 생성 및 출력
for question in questions:
    print('#################################################################')
    answer = qa_chain.invoke({"input": question})

    # 결과에서 'answer' 키를 찾고 출력, 없으면 기본값으로 '답변을 찾을 수 없습니다.' 설정
    response = answer.get('answer', '답변을 찾을 수 없습니다.')
    print(f"질문: {question}")
    print(f"답변: {response}\n")
