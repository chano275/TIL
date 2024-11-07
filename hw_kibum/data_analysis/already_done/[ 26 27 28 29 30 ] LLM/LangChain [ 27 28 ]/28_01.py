import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def process_financial_data():
    # 1. 여러 문서 로드
    documents = []
    for i in range(1, 4):  # 3개의 파일을 로드
        # TODO: TextLoader()를 사용하여 금융 기사 파일을 로드하세요.
        # 이는 금융 기사 파일을 읽어와서 문서를 생성합니다.
        # 이 문제에선 반복문을 사용하기에 "../data/financial_articles_{i}.txt" 라고 적어서 파일을 로드합니다.
        # TextLoader() 참고: https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/
        loader = TextLoader(f"../data/financial_articles_{i}.txt")
        documents.extend(loader.load())

    # 2. 텍스트 분할
    # TODO: RecursiveCharacterTextSplitter()를 사용하여 문서를 작은 청크로 나눕니다.
    # 이는 문서를 작은 단위로 나누어 처리할 수 있도록 합니다.
    # chunk_size는 1000, chunk_overlap은 200으로 설정합니다.
    # chunk_size는 각 청크의 길이를 나타내며, chunk_overlap은 청크 간의 중첩을 나타냅니다.
    # chunk_size=1000은 각 청크가 최대 1000자의 텍스트를 포함하도록 설정하여
    # 메모리 사용을 최적화하고 처리 속도를 높입니다.
    # chunk_overlap=200은 청크 간에 200자의 중첩을 허용하여 문맥을 유지하고
    # 중요한 정보를 잃지 않도록 합니다.
    # RecursiveCharacterTextSplitter() 참고: https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # 3. 임베딩 생성
    # TODO: OpenAIEmbeddings()를 사용하여 문서의 텍스트를 벡터로 변환합니다.
    # 이는 문서를 벡터로 변환하여 유사도 검색을 수행할 수 있도록 합니다.
    # 특정 모델과 API 키를 사용하지 않으면 OpenAIEmbeddings()의 매개변수를 기본값으로 설정합니다.
    # OpenAIEmbeddings() 참고: https://python.langchain.com/docs/integrations/text_embedding/openai/
    embeddings = OpenAIEmbeddings()

    # 4. 벡터 저장소 생성
    # TODO: Chroma.from_documents()를 사용하여 임베딩된 벡터를 저장소에 저장합니다.
    # 이는 문서의 벡터를 저장소에 저장하여 유사도 검색을 수행할 수 있도록 합니다.
    # Chroma 클래스는 벡터 저장소를 생성하고 관리하는 역할을 합니다.
    # from_documents() 메서드는 주어진 문서와 임베딩을 사용하여 벡터 저장소를 초기화합니다.
    # documents 매개변수는 임베딩할 텍스트 데이터(splits)를 제공하며, embedding 매개변수는 텍스트를 벡터로 변환하는 임베딩 모델(embeddings)을 지정합니다.
    # Chroma 클래스 참고: https://python.langchain.com/docs/integrations/vectorstores/chroma/
    # vector store, from_documents() 참고: https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    """
    <FAISS vs Chroma>
    FAISS와 Chroma는 모두 벡터 검색을 위한 라이브러리이지만, 각각의 장단점이 있습니다.
     FAISS의 장점:
     - 성능: FAISS(Facebook AI Similarity Search)는 밀집 벡터의 빠른 유사도 검색과 클러스터링에 최적화되어 있습니다.
     - 확장성: 대규모 데이터셋을 효율적으로 처리할 수 있습니다.
     - 유연성: 속도와 정확성 간의 균형을 맞추기 위해 다양한 인덱싱 방법과 구성을 지원합니다.
     FAISS의 단점:
     - 복잡성: 다양한 인덱싱 방법과 구성을 이해하는 데 더 많은 노력이 필요합니다.
     - 종속성: 추가적인 종속성과 설정이 필요하며, 일부 환경에서는 복잡할 수 있습니다.
     Chroma의 장점:
     - 사용 용이성: Chroma는 사용자가 쉽게 사용할 수 있도록 설계되었습니다.
     - 통합성: Langchain의 문서 처리 및 임베딩 워크플로우와 원활하게 통합됩니다.
     - 단순성: 벡터 저장소를 생성하고 관리하기 위한 간단한 API를 제공합니다.
     Chroma의 단점:
     - 성능: FAISS만큼 대규모 또는 고성능 시나리오에 최적화되어 있지 않을 수 있습니다.
     - 유연성: FAISS에 비해 인덱싱 방법과 구성 옵션이 적습니다.
    """

    # 5. 유사도 검색 수행
    query = "주요 금융 지표는 무엇인가요?"
    # TODO: vectorstore.simularity_search()를 사용하여 유사도 검색을 수행합니다
    # similarity_search() 메서드는 주어진 쿼리(query)와 벡터 저장소(vectorstore)에 저장된 문서 벡터 간의 유사도를 계산하여
    # 가장 유사한 문서를 반환합니다.
    # 상위 3개의 유사한 문서를 검색하려면 k 매개변수를 3으로 설정합니다.
    # similarity_search() 메서드 참고: https://python.langchain.com/docs/integrations/vectorstores/chroma/#similarity_search
    results = vectorstore.similarity_search(query, k=3)

    return results

# 메인 실행 부분
if __name__ == "__main__":
    results = process_financial_data()
    print("\n" + "=" * 70)
    print("금융 데이터 처리 결과")
    print("=" * 70)
    print("\n'주요 금융 지표는 무엇인가요?'에 대한 질문과 관련 있는 데이터 순위별 나열\n")
    for i, doc in enumerate(results, 1):
        file_name = os.path.basename(doc.metadata['source'])
        print(f"{i}순위: {file_name}")
        print("-" * 60)
        print(f"내용: {doc.page_content[:200]}...")
        print("-" * 60)
        print()


#######################################################################
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "your-api-key-here"


def optimize_langchain_process():
    # 1. 문서 로드 및 분할
    # TextLoader를 사용하여 금융 기사 파일을 로드합니다.
    # 이는 금융 기사 파일을 읽어와서 문서를 생성합니다.
    # TextLoader() 참고: https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/
    loader = TextLoader("../data/financial_articles.txt")
    documents = loader.load()

    # RecursiveCharacterTextSplitter를 사용하여 문서를 작은 청크로 나눕니다.
    # chunk_size=500은 각 청크가 최대 500자의 텍스트를 포함하도록 설정하여
    # 메모리 사용을 최적화하고 처리 속도를 높입니다.
    # chunk_overlap=50은 청크 간 50자의 중첩을 허용하여 문맥을 유지하고
    # 중요한 정보를 잃지 않도록 합니다.
    # 과제1과 비교하여 더 작은 chunk_size와 chunk_overlap을 사용한 이유는 다음과 같습니다:
    # 1. 최적화 목적: 더 세밀한 텍스트 분할을 시도하여 최적화를 목표로 합니다.
    # 2. 검색 정확도 향상: 더 작은 텍스트 단위로 분할하여 검색의 정확도를 높입니다.
    # RecursiveCharacterTextSplitter() 참고: https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)

    # 2. 벡터 저장소 생성
    # OpenAIEmbeddings를 사용하여 문서의 텍스트를 벡터로 변환합니다.
    # 이는 문서를 벡터로 변환하여 유사도 검색을 수행할 수 있도록 합니다.
    # OpenAIEmbeddings() 참고: https://python.langchain.com/docs/integrations/text_embedding/openai/
    embeddings = OpenAIEmbeddings()

    # Chroma.from_documents를 사용하여 임베딩된 벡터를 저장소에 저장합니다.
    # 이는 문서의 벡터를 저장소에 저장하여 유사도 검색을 수행할 수 있도록 합니다.
    # Chroma 클래스는 벡터 저장소를 생성하고 관리하는 역할을 합니다.
    # from_documents() 메서드는 주어진 문서와 임베딩을 사용하여 벡터 저장소를 초기화합니다.
    # documents 매개변수는 임베딩할 텍스트 데이터(splits)를 제공하며, embedding 매개변수는 텍스트를 벡터로 변환하는 임베딩 모델(embeddings)을 지정합니다.
    # Chroma 클래스 참고: https://python.langchain.com/docs/integrations/vectorstores/chroma/
    # vector store, from_documents() 참고: https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # 3. 메모리 설정
    # TODO: ConversationBufferMemory()를 사용하여 대화 기록을 저장하는 메모리를 생성하세요.
    # ConversationBufferMemory를 사용하여 대화 기록을 저장하는 메모리를 생성합니다.
    # memory_key="chat_history"는 대화 기록이 "chat_history"라는 키로 저장됨을 나타냅니다.
    # return_messages=True는 메모리가 전체 대화 기록을 메시지 목록으로 반환하도록 설정합니다.
    # ConversationBufferMemory() 참고: https://python.langchain.com/v0.1/docs/modules/memory/types/buffer/
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 4. 프롬프트 템플릿 설정
    # PromptTemplate을 사용하여 프롬프트 템플릿을 생성합니다.
    prompt_template = """당신은 금융 전문가입니다. 주어진 컨텍스트를 바탕으로 다음 질문에 대해 간결하고 정확하게 답변해주세요:

    컨텍스트: {context}

    질문: {question}

    가능한 한 객관적이고 사실에 기반한 답변을 제공해주세요.
    """
    # TODO: PromptTemplate()을 사용하여 프롬프트 템플릿을 생성하세요.
    # PromptTemplate을 사용하여 프롬프트 템플릿을 생성합니다.
    # template 매개변수는 프롬프트의 형식(prompt_template)을 지정합니다.
    # input_variables 매개변수는 프롬프트에서 사용할 변수 목록(["context", "question"])을 지정합니다.
    # PromptTemplate() 참고: https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.prompt.PromptTemplate.html
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # 5. 대화형 검색 체인 생성
    # ConversationalRetrievalChain.from_llm을 사용하여 대화형 검색 체인을 생성합니다.
    # llm 매개변수는 ChatOpenAI를 사용하여 설정하고, retriever 매개변수는 vectorstore.as_retriever()를 사용하여 설정합니다.
    # memory 매개변수는 대화 기록을 저장하는 메모리를 지정하고,
    # combine_docs_chain_kwargs 매개변수는 프롬프트 템플릿을 지정합니다.
    # ConversationalRetrievalChain 참고: https://python.langchain.com/api_reference/langchain/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html
    # from_llm() 메서드 참고: https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    return chain

# 메인 실행 부분
if __name__ == "__main__":
    chain = optimize_langchain_process()

    questions = [
        "최근 금융 시장의 주요 동향은 무엇인가요?",
        "금리 인상이 경제에 미치는 영향에 대해 설명해주세요.",
        "이전 질문들의 내용을 고려하여, 현재 투자자들이 주의해야 할 점은 무엇인가요?"
    ]

    for i, question in enumerate(questions, 1):
        result = chain({"question": question})
        print(f"\n질문 {i}: {question}")
        print(f"답변 {i}: {result['answer']}")
        print("-" * 50)