# [ LangChain으로 간단한 LLM 챗봇 만들기 ]

"""
실습 목표 : - LangChain을 활용해서 gpt-4o-mini 모델을 사용하는 챗봇을 개발
           - 짧은 Chain을 구성하고, 이를 활용해서 챗봇을 구현

실습 목차 :
1. ChatOpenAI Agent 생성 : 사용자의 입력에 대한 ChatGPT의 gpt-4o-mini 모델의 답변을 받아오는 Agent를 생성
2. 챗봇 Chain 구성        : ChatOpenAI Agent를 비롯하여 챗봇 구현에 필요한 Agent들을 엮어서 챗봇 Chain으로 구성
3. 챗봇 사용              : 여러분이 구성하신 챗봇을 사용
"""

# TODO: 환경 설정
from langchain.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama, ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
import os

if "OPENAI_API_KEY" not in os.environ:os.environ["OPENAI_API_KEY"] = ''

# TODO: ChatOpenAI Agent 생성
"""
gpt-4o-mini 모델을 사용하는 ChatOpenAI Agent를 생성
- ChatOpenAI Agent는 사용자의 입력을 Ollama를 통해 로컬에서 구동한 LLM에 전송하고, 그 답변을 반환
- 본 RAG 과정에서는 LLM으로 ChatOpenAI를 활용할 것
"""
llm = ChatOpenAI(model="gpt-4o-mini", api_key="")  # gpt-4o-mini 모델을 사용하는 ChatOpenAI 객체 생성



# TODO: ChatOpenAI Agent 사용
"""
Agent를 구성했으니, 이제 Agent를 사용해봅시다.
# 1-1. Runnable interface
LangChain에서 Chain으로 엮을 수 있는 대부분의 구성 요소 (Agent, Tool 등..)는 "Runnable" protocol을 공유
- 관련 LangChain API 문서: [langchain_core.runnables.base.Runnable — 🦜🔗 LangChain 0.1.16](https://api.python.langchain.com/en/stable/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable)

Runnable protocol을 공유하는 구성 요소들이 가지고 있는 3가지 메서드 : 
- stream: 구성 요소의 답변을 순차적으로 반환 (stream back)
- invoke: 입력된 값으로 chain을 호출하고, 그 결과를 반환
- batch: 입력값 리스트 (batch)로 chain을 호출하고, 그 결과를 반환

방금 사용한 `ChatOpenAI` Class : "Runnable" 하기 때문에 `invoke` 메서드를 가짐
- invoke() 메서드를 통해 Agent, Chain 등에 데이터를 입력 => 그 출력을 받아오는 형식

"당신은 누구입니까?" 라는 질문을 입력 => Agent가 OpenAI API를 통해 Mistral 7B 모델의 답변을 받아 출력할 것
단순 텍스트 뿐만 아니라, 시스템, 사람, AI의 답변을 리스트로 정리하여 입력 가능
여기서는 LangChain의 `SystemMessage`, `HumanMessage` Class를 활용
"""
# content='저는 OpenAI에서 개발한 AI 언어 모델입니다. 다양한 질문에 답하고, 정보 제공, 글쓰기, 대화 등을 도와드릴 수 있습니다. 무엇을 도와드릴까요?' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 43, 'prompt_tokens': 14, 'total_tokens': 57, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_f59a81427f', 'finish_reason': 'stop', 'logprobs': None} id='run-178b0c55-b176-4401-9ae0-3e18357721a0-0'
# print(llm.invoke("당신은 누구입니까?")) # 'invoke` 메서드 사용

# messages = [SystemMessage("당신은 친절한 AI 어시스턴트 입니다."), HumanMessage("당신을 소개해주세요."),]  # 시스템 프롬프트에 '친절한 AI 어시스턴트' 라는 역할을 명시
# response = llm.invoke(messages)                                                                   # 이제 gpt-4o-mini 모델이 아까와 같은 질문에 어떻게 답했는지 확인

# content='안녕하세요! 저는 AI 어시스턴트입니다. 다양한 주제에 대해 정보를 제공하고 질문에 답하며, 여러분의 필요에 맞는 도움을 드리기 위해 여기 있습니다. 언어나 문화, 과학, 기술 등 여러 분야에 대한 질문을 해주시면 최선을 다해 도와드리겠습니다. 무엇을 도와드릴까요?' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 76, 'prompt_tokens': 31, 'total_tokens': 107, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_f59a81427f', 'finish_reason': 'stop', 'logprobs': None} id='run-a8a7c0e2-c13f-4db7-85dc-404c73da64e5-0'
# print(response)  # 조금 더 친절해짐.


# TODO: 챗봇 Chain 구성 :
"""
위의 `llm` object의 반환 값 > 다른 챗봇을 쓸 때 처럼 답변만 출력 X / 다양한 메타 데이터 까지 같이 출력
ChatGPT를 쓸 때를 생각해보면, 챗봇에 이걸 그대로 출력하는건 좀 부자연 > 이를 방지하기 위해, 답변을 parsing하는 `StrOutputParser` 사용

### 2-1. Output Parser : ChatOpenAI Agent를 비롯하여 LLM 답변 중 [ content만 자동으로 추출 ] 하는 Tool인 `StrOutputParser` 사용
"""
# role = "국토교통부 직원"
# messages = [SystemMessage(f"당신은 {role} 입니다."), HumanMessage("당신을 소개해주세요."),]
# response = llm.invoke(messages)
# content='안녕하세요! 저는 국토교통부에서 근무하는 직원입니다. 제 역할은 국토와 교통 관련 정책을 수립하고, 이를 통해 국민의 안전하고 편리한 이동을 지원하는 것입니다. 또한, 다양한 교통 인프라 개발과 도시 계획, 주택 정책 등에도 관여하고 있습니다. 궁금한 사항이나 도움이 필요하시면 언제든지 질문해 주세요!' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 85, 'prompt_tokens': 28, 'total_tokens': 113, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_f59a81427f', 'finish_reason': 'stop', 'logprobs': None} id='run-1d76893b-442d-4e16-bcd6-c515fca16760-0'
# print(response)


parser = StrOutputParser()
# parsed_response = parser.invoke(response)  # Parser가 제대로 답변만을 리턴하는지 확인
# response에서 의도한 대로 텍스트만 추출하는 것을 확인 가능 - 안녕하세요! 저는 국토교통부의 직원으로, 교통, 국토 개발, 주택 정책 등 다양한 분야에 대한 정보를 제공하고 지원하는 역할을 맡고 있습니다. 국민의 안전하고 편리한 교통수단을 제공하고, 지속 가능한 국토 개발을 위해 노력하고 있습니다. 궁금한 점이나 필요하신 정보가 있다면 언제든지 말씀해 주세요!
# print(parsed_response)



# TODO: 간단한 체인 구성 :
"""
`ChatOpenAI` 를 통해 gpt-4o-mini 모델의 답변을 받았고, 그 받은 답변을 다시 `StrOutputParser`에 입력해서 답변만 추출했음 
                         이 과정을 Chain으로 엮어서 간략화

Chain 역시 "Runnable" > `invoke` 메서드를 통해 Chain의 각 구성요소의 `invoke` 메서드를 순차적으로 호출 가능
이때 특정 객체의 `invoke` 반환값 : Chain 상에서 연결된 다음 객체의 `invoke` 메서드에 입력
체인을 실행 => 체인에 포함된 모든 객체가 순차적으로 실행 => 마지막 객체의 결과 반환

여기서는 llm 객체가 먼저 실행 => 그 결과가 parser 객체에 전달
"""
chain = llm | parser                        # pipe (|) 연산자를 통해 두 객체를 연결해서 하나의 체인으로
chained_response = chain.invoke(messages)
print(chained_response)                     # 별도의 절차 없이 바로 답변만 생성되는 것을 확인 가능


# TODO: 프롬프트 템플릿 :
"""
챗봇에 프로그래밍 조수, 시장조사 요원, 그냥 친구 등 다양한 역할을 적용해야 하는 상황이라 가정
구현 가능한 방법은 여러가지가 있지만, 우선 가장 간단한 방법으로 시스템 프롬프트에 '당신은 {역할} 입니다' 를 입력
이 방법이 항상 잘 작동하는 것은 아니지만, 간단한 예시는 구현 가능
사용자의 입력을 받고, 그에 대응하는 답변을 하기 위해서는 사용자의 입력을 적용할 수 있는 프롬프트 템플릿을 적용 가능

role에는 "AI 어시스턴트"가, 
question에는 "당신을 소개해주세요."       넣을 수 있음 

** Note. 사용한 문자열 : f-string X  =>  중괄호로 감싼 텍스트 : [ LangChain placeholder ] 나타내는 문자열

앞서 저희가 정의했던 코드와 크게 두가지 차이점 존재
- HumanMessage, SystemMessage 같은게 없고, 튜플에 역할과 프롬프트가 저장되어 있음
- 프롬프트에 {question} 같은 placeholder가 존재
"""
messages_with_variables = [("system", "당신은 {role} 입니다."), ("human", "{question}"),]
prompt = ChatPromptTemplate.from_messages(messages_with_variables)
chain = prompt | llm | parser  # pipe (|) 연산자를 통해 여러 객체를 연결해서 하나의 체인으로 만들 수 있습니다.
# prompt 객체 통해 변수를 적용한 프롬프트 생성 > llm 객체 통해 이 프롬프트를 실행 > parser 객체를 통해 결과를 파싱
chain.invoke({"role": "국토 교통부 직원 김싸피", "question": "당신을 소개해주세요."})


# TODO: 챗봇 사용
"""
1. 사용자의 입력을 받아 앞서 정의한 Chain을 실행하고, 그 결과를 반환하는 함수를 정의합니다.
# 간단한 실습이므로 앞서 사용했던 변수를 그대로 함수의 파라미터로 설정했습니다. 
# 다음 실습 부터는 이를 좀 더 고도화 해 볼 것입니다.
"""
def simple_chat(role, question, chain):
    result = chain.invoke({"role": role, "question": question})
    return result

role = input("제 역할을 입력해주세요: ")
while True:
    question = input("질문을 입력해주세요 (역할을 바꾸고 싶다면 '역할 교체' 를 입력해주세요. 종료를 원하시면 '종료'를 입력해주세요.): ")
    if question == "역할 교체":
        role = input("역할을 입력해주세요: ")
        continue
    elif question == "종료":
        break
    else:
        # chain = prompt | llm | parser
        result = simple_chat(role, question, chain)
        print(result)

"""
대부분의 경우, 입력한 역할에 맞춰 어느 정도 대답하는 것을 확인할 수 있습니다. <br>
현재 챗봇은 다음 한계점이 있습니다.
- 문서나 데이터 기반 추론이 불가능하다.
- Chat History를 기억하지 못한다.

이어지는 실습에서 두 한계를 개선하고, 교통 3대 혁신 전략 문서 기반 QA 봇을 만들어 봅시다.
"""



# TODO: 챗봇 생성 - PDF
"""
실습 목표

[실습3] RAG를 위한 Vector Score, Retriever 에서 학습한 내용을 바탕으로 LangChain을 활용해서 입력된 문서를 요약해서 Context로 활용하는 챗봇을 개발합니다.

## 실습 목차
---

1. **교통 3대 혁신 전략 문서 벡터화:** RAG 챗봇에서 활용하기 위해 교통 3대 혁신 전략 파일을 읽어서 벡터화하는 과정을 실습합니다.

2. **RAG 체인 구성:** 이전 실습에서 구성한 미니 RAG 체인을 응용해서 간단한 교통 3대 혁신 전략 문서 기반 RAG 체인을 구성합니다.

3. **챗봇 구현 및 사용:** 구성한 RAG 체인을 활용해서 교통 3대 혁신 전략 문서 기반 챗봇을 구현하고 사용해봅니다.

## 실습 개요
---
RAG 체인을 활용해서 교통 3대 혁신 전략 문서 기반 챗봇을 구현하고 사용해봅니다.




## 1. 문서 벡터화
- RAG 챗봇에서 활용하기 위해 교통 3대 혁신 전략 파일을 읽어서 벡터화하는 과정을 실습합니다.
"""
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")