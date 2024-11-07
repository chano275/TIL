import openai
import json
import os

# 경고: API 키를 코드에 직접 포함하는 것은 보안상 위험합니다.
# 실제 프로덕션 환경에서는 환경 변수나 안전한 비밀 관리 시스템을 사용하세요.
API_KEY = "your-api-key"

# OpenAI API 클라이언트 초기화
# API_KEY를 사용하여 OpenAI API client 객체를 생성합니다.
# 이 client 객체를 통해 OpenAI의 다양한 AI 모델과 상호작용할 수 있습니다.
client = openai.OpenAI(api_key=API_KEY)

# 텍스트 생성 함수
# 프롬프트를 입력받아 OpenAI API를 사용해 텍스트를 생성합니다.
# prompt: AI에 전달할 텍스트 프롬프트
# max_tokens: 생성할 텍스트의 최대 토큰 수
# temperature: 생성된 텍스트의 다양성을 조절하는 매개변수(0에 가까울수록 일반적인 출력, 1에 가까울수록 더 다양한 출력)
def generate_financial_text(prompt, max_tokens=300, temperature=0.7):
    """
    OpenAI API를 사용하여 주어진 프롬프트에 기반한 금융 텍스트를 생성합니다.
    """

    # TODO: client.chat.completions.create() 메서드를 사용하여 텍스트를 생성하세요.
    # client.chat.completions.create() 메서드를 사용하여 텍스트를 생성합니다.
    # model, messages, max_tokens, temperature 매개변수를 설정해야 합니다.
    # model은 "gpt-3.5-turbo"를 사용합니다.
    # messages는 system과 user 역할을 사용하여,
    # system 역할에서 "당신은 경제 분석가입니다. 한국어로 자세하고 정확한 경제 보고서를 작성해주세요."라는 메시지를 전달하고,
    # user 역할에서 prompt를 전달합니다.
    # max_tokens는 300을 사용하고, temperature는 0.7을 사용하기에 기존에 정의했던 매개변수를 사용합니다.
    # client.chat.completions.create() 참고: https://platform.openai.com/docs/api-reference/chat/create
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 경제 분석가입니다. 한국어로 자세하고 정확한 경제 보고서를 작성해주세요."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )

    # API 응답에서 생성된 텍스트를 추출하고 앞뒤 공백을 제거합니다.
    # 문자열 메서드 참고: https://docs.python.org/3/library/stdtypes.html#string-methods
    return response.choices[0].message.content.strip()

# 데이터 저장 함수
# 데이터를 JSON 파일로 저장합니다.
def save_to_json(data, filename):
    """
    주어진 데이터를 JSON 형식으로 파일에 저장합니다.
    """
    # open() 함수 참고: https://docs.python.org/3/library/functions.html#open
    with open(filename, 'w', encoding='utf-8') as f:
        # data를 파일 f에 저장하고, ensure_ascii=False, indent=4 옵션을 사용합니다.
        # ensure_ascii=False는 유니코드 문자를 ASCII로 인코딩하지 않고 그대로 저장하도록 합니다.
        # indent=4는 들여쓰기를 사용하여 JSON 파일을 읽기 쉽도록 만듭니다.
        # json.dump() 참고: https://docs.python.org/3/library/json.html#json.dump
        json.dump(data, f, ensure_ascii=False, indent=4)

# 품질 평가 함수
# 생성된 텍스트의 품질을 평가하는 함수를 구현합니다.
def evaluate_text_quality(text):
    """
    생성된 텍스트의 품질을 평가합니다.
    """

    # TODO: 생성된 텍스트의 품질을 평가하는 함수를 구현하세요.
    # in 연산자를 사용하여 'GDP'나 '성장률' 키워드가 포함되어 있는지 확인합니다.
    # 'GDP'나 '성장률' 키워드가 포함되어 있는지 확인해서 구체적인 정보 포함 여부를 판단합니다.
    # in 연산자 참고: https://docs.python.org/3/reference/expressions.html#membership-test-operations
    has_details_1 = "GDP" in text

    has_details_2 = "성장률" in text

    return {
        "has_details_1": has_details_1,
        "has_details_2": has_details_2
    }

# 텍스트 생성에 사용할 프롬프트 리스트
# 각 튜플은 (프롬프트 이름, 프롬프트 내용)으로 구성됩니다.
prompts = [
    ("simple", "2024년 한국 경제 전망에 대한 간략한 보고서를 100단어 내외로 작성해주세요."),
    ("detailed", "2024년 세계 경제 전망에 대해 자세히 논의해주세요. GDP 성장률, 인플레이션, 주요 국가(미국, 중국, EU)의 경제 정책, 주요 산업의 성장 전망 등을 포함하여 300단어 내외로 작성해주세요.")
]

# 각 프롬프트에 대해 텍스트를 생성하고 평가합니다.
for name, prompt in prompts:

    # 텍스트 생성
    generated_text = generate_financial_text(prompt, max_tokens=500 if name == "detailed" else 200)
    # 생성된 텍스트를 JSON 파일로 저장
    save_to_json({"prompt": prompt, "text": generated_text}, f"{name}_output.json")

    # 생성된 텍스트의 품질 평가
    evaluation = evaluate_text_quality(generated_text)
    # 평가 결과 출력
    print(f"\n=== {name} 프롬프트 텍스트 품질 평가 ===")
    print("구체적인 정보(GDP) 포함 여부 :", "예" if evaluation["has_details_1"] else "아니요")
    print("구체적인 정보(성장률) 포함 여부 :", "예" if evaluation["has_details_2"] else "아니요")

    # 생성된 텍스트의 일부 출력
    print("\n생성된 텍스트 (첫 100자):")
    print(generated_text[:100] + "...")


##############################################################


import openai
import json
import os

# 경고: API 키를 코드에 직접 포함하는 것은 보안상 위험합니다.
# 실제 프로덕션 환경에서는 환경 변수나 안전한 비밀 관리 시스템을 사용하세요.
API_KEY = "your-api-key"

# OpenAI API 클라이언트 초기화
# API_KEY를 사용하여 OpenAI API client 객체를 생성합니다.
# 이 client 객체를 통해 OpenAI의 다양한 AI 모델과 상호작용할 수 있습니다.
client = openai.OpenAI(api_key=API_KEY)

# 텍스트 생성 함수
# 프롬프트를 입력받아 OpenAI API를 사용해 텍스트를 생성합니다.
def generate_financial_text(prompt, max_tokens=300, temperature=0.7):
    """
    OpenAI API를 사용하여 주어진 프롬프트에 기반한 금융 텍스트를 생성합니다.
    """

    # client.chat.completions.create() 메서드를 사용하여 텍스트를 생성합니다.
    # client.chat.completions.create() 참고: https://platform.openai.com/docs/api-reference/chat/create
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 경제 분석가입니다. 한국어로 자세하고 정확한 경제 보고서를 작성해주세요."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )

    # API 응답에서 생성된 텍스트를 추출하고 앞뒤 공백을 제거합니다.
    # 문자열 메서드 참고: https://docs.python.org/3/library/stdtypes.html#string-methods
    return response.choices[0].message.content.strip()

# 최적화된 텍스트 생성 함수
def generate_optimized_text(prompt, max_tokens=150, temperature=0.5):
    """
    최적화된 매개변수를 사용해 텍스트를 생성합니다.
    """
    # TODO: 최적화된 텍스트 생성을 위한 로직을 구현하세요.
    # client.chat.completions.create() 메서드를 사용하여 텍스트를 생성합니다.
    # model. message, max_tokens, temperature 매개변수를 최적화된 값으로 설정합니다.
    # model은 기존대로 "gpt-3.5-turbo"를 사용합니다.
    # messages는 system과 user 역할을 사용하여, system 역할에서
    # content 값으로 "당신은 간결하고 정확한 경제 분석가입니다. 핵심만을 간단히 요약해주세요."를 전달하고,
    # user 역할에서 prompt를 전달합니다.
    # max_tokens는 150을 사용하고, temperature는 0.5를 사용하기에 기존에 정의했던 매개변수를 사용합니다.
    # client.chat.completions.create() 참고: https://platform.openai.com/docs/api-reference/chat/create
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 간결하고 정확한 경제 분석가입니다. 핵심만을 간단히 요약해주세요."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

# 데이터 저장 함수
# 데이터를 JSON 파일로 저장합니다.
def save_to_json(data, filename):
    """
    주어진 데이터를 JSON 형식으로 파일에 저장합니다.
    """
    # open() 함수 참고: https://docs.python.org/3/library/functions.html#open
    with open(filename, 'w', encoding='utf-8') as f:
        # data를 파일 f에 저장하고, ensure_ascii=False, indent=4 옵션을 사용합니다.
        # ensure_ascii=False는 유니코드 문자를 ASCII로 인코딩하지 않고 그대로 저장하도록 합니다.
        # indent=4는 들여쓰기를 사용하여 JSON 파일을 읽기 쉽도록 만듭니다.
        # json.dump() 참고: https://docs.python.org/3/library/json.html#json.dump
        json.dump(data, f, ensure_ascii=False, indent=4)

# 품질 평가 함수
# 생성된 텍스트의 품질을 평가하는 함수를 구현합니다.
def evaluate_text_quality(text):
    """
    생성된 텍스트의 품질을 평가합니다.
    """
    # in 연산자를 사용하여 'GDP'나 '성장률' 키워드가 포함되어 있는지 확인합니다.
    # 'GDP'나 '성장률' 키워드가 포함되어 있는지 확인해서 구체적인 정보 포함 여부를 판단합니다.
    # in 연산자 참고: https://docs.python.org/3/reference/expressions.html#membership-test-operations
    has_details_1 = "GDP" in text

    has_details_2 = "성장률" in text

    return {
        "has_details_1": has_details_1,
        "has_details_2": has_details_2
    }

# 텍스트 생성에 사용할 프롬프트 리스트
# 각 튜플은 (프롬프트 이름, 프롬프트 내용)으로 구성됩니다.
prompts = [
    ("simple", "2024년 한국 경제 전망에 대한 간략한 보고서를 100단어 내외로 작성해주세요."),
    ("detailed", "2024년 세계 경제 전망에 대해 자세히 논의해주세요. GDP 성장률, 인플레이션, 주요 국가(미국, 중국, EU)의 경제 정책, 주요 산업의 성장 전망 등을 포함하여 300단어 내외로 작성해주세요.")
]

# 각 프롬프트에 대해 텍스트 생성, 최적화, 평가를 수행합니다.
for name, prompt in prompts:
    # 원본 텍스트 생성
    original_text = generate_financial_text(prompt, max_tokens=500 if name == "detailed" else 200)
    save_to_json({"prompt": prompt, "text": original_text}, f"{name}_output.json")

    # 최적화된 텍스트 생성
    optimized_text = generate_optimized_text(prompt)
    save_to_json({"prompt": prompt, "optimized_text": optimized_text}, f"{name}_optimized_output.json")

    # 품질 평가
    original_evaluation = evaluate_text_quality(original_text)
    optimized_evaluation = evaluate_text_quality(optimized_text)

    # 결과 출력
    print(f"\n=== {name} 프롬프트 최적화 전후 비교 ===")
    print("구체적인 정보(GDP) 포함 여부 (최적화 전):", "예" if original_evaluation["has_details_1"] else "아니요")
    print("구체적인 정보(GDP) 포함 여부 (최적화 후):", "예" if optimized_evaluation["has_details_1"] else "아니요")
    print("구체적인 정보(성장률) 포함 여부 (최적화 전):", "예" if original_evaluation["has_details_2"] else "아니오")
    print("구체적인 정보(성장률) 포함 여부 (최적화 후):", "예" if optimized_evaluation["has_details_2"] else "아니오")

    print("\n원본 텍스트 (첫 100자):")
    print(original_text[:100] + "...")
    print("\n최적화된 텍스트 (첫 100자):")
    print(optimized_text[:100] + "...")