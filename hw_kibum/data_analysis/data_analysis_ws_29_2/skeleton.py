from langchain_core.prompts import PromptTemplate

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

print(prompt)
