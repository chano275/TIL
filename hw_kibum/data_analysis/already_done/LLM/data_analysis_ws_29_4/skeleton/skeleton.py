import json
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader

api_key = "YOUR_API_KEY"

# PDF 파일에서 텍스트 조각을 추출하는 함수입니다.
# 이 함수는 문서를 정교하게 분할하기보다는, 데이터를 생성하기 위해 간단히 청킹작업을 수행하기 위한 목적으로 사용합니다.
def extract_pdf_elements(filepath):
    """
    LangChain의 PyMuPDFLoader를 사용하여 PDF 파일에서 텍스트 조각을 추출합니다.
    """
    # langchain_community.document_loaders 모듈의 PyMuPDFLoader 클래스는 PyMuPDF를 사용하여 PDF 파일의 페이지를 로드하고, 
    # 각 페이지를 개별 document 객체로 추출합니다. 
    # 특히 PDF 문서의 자세한 메타데이터를 추출하는 데 강점이 있습니다.
    # TODO: PyMuPDFLoader 객체를 통해 document 추출을 완성하세요.
    loader = ______________
    documents = ______________  # PDF를 LangChain Document 형식으로 로드합니다.
    return documents

# PDF 파일 로드 및 텍스트 추출
elements = extract_pdf_elements("../data/canada_av-68-90.pdf")

#################################################################################################################################

# LangChain을 사용한 프롬프트 템플릿 정의
# "context": element.page_content, "domain": "vehicle", "num_questions": "3"으로 적용됩니다.
# TODO: 해당 예시를 기반으로 문서를 기반으로 아래의 예시와와 같은 형태의 질답 형태를 만들 수 있게 prompt 객체에 프롬프팅하세요. 
# 결과는 원하는 바와 맞게 잘 출력되기만 하면 문제없습니다!
"""
[예시]
```json
{{
    "QUESTION": "최근 발표된 자율주행 자동차 규제의 주요 목표는 무엇입니까?",
    "ANSWER": "최근 발표된 자율주행 자동차 규제의 주요 목표는 보행자 안전 강화와 자율주행 시스템의 신뢰성 확보입니다."
}},
{{
    "QUESTION": "테슬라의 자율주행 시스템은 어떤 조건에서 완전 자율주행을 목표로 하고 있습니까?",
    "ANSWER": "테슬라의 자율주행 시스템은 고속도로와 같은 제한된 조건에서 완전 자율주행을 목표로 하고 있습니다."
}},
{{
    "QUESTION": "2025년까지 글로벌 자율주행 차량 시장의 성장 전망은 어떻게 예측되고 있습니까?",
    "ANSWER": "2025년까지 글로벌 자율주행 차량 시장은 연평균 23% 성장할 것으로 예측되고 있습니다."
}}
해당 예시를 통해 {context}, {domain}, {num_questions}에 맞게 위와 같은 형태의 결과를 낼 수 있는 프롬프트
"""


prompt = PromptTemplate.from_template(
    """아래는 배경 정보입니다. 이 배경 정보 외에는 아무 것도 모릅니다.
---------------------

{context}

---------------------

#형식:
```json
{{
    "QUESTION": "최근 발표된 자율주행 자동차 규제의 주요 목표는 무엇입니까?",
    "ANSWER": "최근 발표된 자율주행 자동차 규제의 주요 목표는 보행자 안전 강화와 자율주행 시스템의 신뢰성 확보입니다."
}},
{{
    "QUESTION": "테슬라의 자율주행 시스템은 어떤 조건에서 완전 자율주행을 목표로 하고 있습니까?",
    "ANSWER": "테슬라의 자율주행 시스템은 고속도로와 같은 제한된 조건에서 완전 자율주행을 목표로 하고 있습니다."
}},
{{
    "QUESTION": "2025년까지 글로벌 자율주행 차량 시장의 성장 전망은 어떻게 예측되고 있습니까?",
    "ANSWER": "2025년까지 글로벌 자율주행 차량 시장은 연평균 23% 성장할 것으로 예측되고 있습니다."
}}
"""
)

#################################################################################################################################

# 응답 JSON 데이터를 파싱하는 함수
def custom_json_parser(response):
    # 응답 텍스트에서 JSON 구문을 제거하고 JSON 문자열로 변환합니다.
    json_string = response.content.strip().removeprefix("```json\n").removesuffix("\n```").strip()
    json_string = f'[{json_string}]'  # JSON 배열로 변환
    return json.loads(json_string)  # JSON 형식의 데이터를 파이썬 객체로 변환하여 반환

# LangChain을 사용하여 질문-답변 생성 체인 구성
# TODO: chain을 구성할 때는 LCEL(Langchain expression language)를 활용해서 구성해주세요.
# LCEL : Langchain에서 제공하는 기능들을 조합한 Chain을 마치 블록처럼 쉽게 분해, 조립할 수 있도록 설계한 프레임워크
# chain = prompt | model | output_parser와 같이 "|"를 통해 흐름을 엮는 것으로 이해할 수 있습니다.
chain = (
    ______________  # 프롬프트 템플릿 설정
    | ______________(
        model="gpt-4o-mini",  # 모델 설정
        temperature=0,  # 출력의 다양성을 낮추는 온도 설정
        streaming=False,  # 스트리밍 옵션
        callbacks=[StreamingStdOutCallbackHandler()],  # 출력 스트리밍 콜백 설정
        api_key=api_key, # GPT사용을 위한 API KEY
    )
    | ______________  # 기존에 만든 JSON 파싱 함수 설정
)

# 질문-답변을 저장할 리스트 초기화
qa_pair = []

# PDF 텍스트 청크를 순회하여 질문과 답변 생성
for element in elements:
    if element.page_content:
        qa_pair.extend(
            chain.invoke(
                _________________________________________ 
                # TODO: invoke하기 위해 안에 들어갈 요소를 각각 넣어야 합니다.
                # "context": element.page_content, "domain": "vehicle", "num_questions": "3"으로 적용됩니다.
            )
        )

#################################################################################################################################

# 디버깅을 위한 데이터셋 추가
additional_qa = [
    {
        "QUESTION": "삼성 청년 SW 아카데미(SSAFY)란 무엇인가요?",
        "ANSWER": "삼성 청년 SW 아카데미(SSAFY)는 삼성의 SW 교육 경험과 고용노동부의 취업지원 노하우를 바탕으로 취업 준비생에게 SW 역량 향상 교육 및 다양한 취업지원 서비스를 제공하여 취업에 성공하도록 돕는 프로그램입니다."
    },
    {
        "QUESTION": "SSAFY가 제공하는 교육의 특징은 무엇인가요?",
        "ANSWER": "SSAFY는 최고 수준의 교육을 제공합니다. 전문분야별 자문교수단과 삼성의 SW 전문가가 참여한 명품 커리큘럼을 통해 경쟁력 있는 차세대 SW 인력을 양성합니다."
    },
    {
        "QUESTION": "SSAFY에서 어떤 맞춤형 교육을 받을 수 있나요?",
        "ANSWER": "SSAFY는 개인별 SW 역량 및 이해도 수준, 전공에 따라 맞춤형 교육을 제공합니다. 이를 통해 학습자는 자신에게 최적화된 교육을 받을 수 있어 높은 학습 효과를 기대할 수 있습니다."
    },
    {
        "QUESTION": "SSAFY가 지향하는 학습 방식은 무엇인가요?",
        "ANSWER": "SSAFY는 자기주도적 학습을 지향합니다. 단순히 지식을 전달하는 것이 아니라, 스스로 문제를 해결할 수 있는 능력을 강화시키며, 실무 적응력을 높이기 위해 기업에서 실제로 수행하는 프로젝트를 통해 실습을 제공합니다."
    },
    {
        "QUESTION": "SSAFY에서 제공하는 취업지원 서비스는 무엇인가요?",
        "ANSWER": "SSAFY는 고용노동부의 취업지원 노하우를 바탕으로 최적의 일자리 정보를 제공하며, 취업 실전 교육과 컨설팅 서비스를 통해 교육생의 취업 경쟁력을 높이고 성공적인 취업을 지원합니다."
    }
]

# 원래 생성한 qa_pair와 추가된 qa_pair 결합
qa_pair.extend(additional_qa)

# 결과를 JSONL 파일로 저장
with open("traffic_qa_pair.jsonl", "w", encoding="utf-8") as f:
    for qa in qa_pair:
         # TODO: 각 key에 맞는 값을 넣어주세요.
        qa_modified = {
            "instruction": ________________,  # json에서 불러온 QUESTION 값을 instruction의 값으로 불러오는 작업을 수행
            "input": "",  # 입력은 빈 문자열로 설정
            "output": ________________,  # json에서 불러온 ANSWER 값을 output의 값으로 설정하는 작업을 수행
        }
        f.write(json.dumps(qa_modified, ensure_ascii=False) + "\n")  # JSONL 형식으로 저장

print(f"추가 후 qa_pair의 길이 : {len(qa_pair)}")
