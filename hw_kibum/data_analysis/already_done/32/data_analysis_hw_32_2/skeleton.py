# 과제 1: AI 데이터 생성

# 필요한 라이브러리 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
import random
import json

# OpenAI API Key 설정 - API 키를 환경 변수에 설정하여 OpenAI 서비스를 사용합니다.
os.environ["OPENAI_API_KEY"] = ""

# 임베딩 모델 설정 - 텍스트 데이터를 수치 벡터로 변환하기 위한 임베딩 모델
# OpenAIEmbeddings() 참고: https://python.langchain.com/docs/integrations/text_embedding/openai/#instantiation
# 이 문제에선, OpenAIEmbeddings()를 사용하여 임베딩 모델을 설정합니다.
# TODO: 임베딩 모델을 설정하기 위한 함수를 완성하세요.
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# 교통 관련 시드 키워드 - 각 상황별 데이터를 생성하기 위해 사용될 키워드
traffic_keywords = ["신호등 고장", "차량 정체", "교통사고"]

# 골드 샘플 생성 - 각 상황별로 몇 가지 예시 응답을 사전 정의하여 모델에 제공
gold_samples = [
    {"situation": "신호등 고장", "response": "신호등이 고장났을 때는 교차로에서 일시 정지 후 좌우를 살피고 진행하세요."},
    {"situation": "차량 정체", "response": "차량 정체 시 차선을 유지하고, 안전거리를 확보하세요."},
    {"situation": "교통사고", "response": "교통사고 발생 시 비상등을 켜고 안전한 장소로 이동하세요."},
]


# 데이터 생성 함수
# 이 함수는 주어진 상황에 대해 모델을 사용하여 응답을 생성합니다.
def generate_data(traffic_keywords, gold_samples, max_data=20, k=5):
    """
        OpenAI 모델을 사용하여 각 상황에 맞는 데이터를 생성합니다.

        Parameters:
        - traffic_keywords (list): 교통 상황을 나타내는 키워드 리스트
        - gold_samples (list): 각 상황에 대한 예시 응답 리스트
        - max_data (int): 생성할 최대 응답 개수
        - k (int): 한 상황에 대해 생성할 응답 개수

        Returns:
        - unique_texts (list): 생성된 고유 응답 리스트
    """
    total_generated = 0
    unique_texts = []

    # OpenAI 모델 인스턴스 생성
    # ChatOpenAI() 참고: https://python.langchain.com/docs/integrations/chat/openai/
    # TODO: 모델 인스턴스를 생성하기 위한 함수를 완성하세요.
    model = ChatOpenAI(model="gpt-4o-mini")

    # 각 키워드에 대해 데이터 생성 시작
    for keyword in traffic_keywords:
        if total_generated >= max_data:
            break

        # 각 상황에 대해 예시 응답을 3개 랜덤 선택
        situation_samples = [sample['response'] for sample in gold_samples if sample["situation"] == keyword]
        if len(situation_samples) < 3:
            few_shot_examples = situation_samples
        else:
            few_shot_examples = random.sample(situation_samples, 3)

        # 예시 응답을 모델에 입력할 형식으로 변환
        few_shot_text = "\n".join(few_shot_examples)

        # 모델에 입력할 프롬프트 생성
        prompt = f"""
        다음은 '{keyword}' 상황에 대한 모범 답변들입니다:
        {few_shot_text}

        위의 예시를 참고하여 '{keyword}' 상황에서 적절한 대응 방안을 {k}가지만 작성하세요.
        각 대응 방안은 한 문장으로만 작성하며, 숫자나 순서 표시를 포함하지 마세요.
        """
        generated_texts = []
        # 상황별로 k개의 응답 생성
        while len(generated_texts) != k:
            response = model.invoke(prompt)
            generated = response.content.strip().split('\n')
            generated = [text.strip() for text in generated if text.strip() != ''][:k]
            generated_texts += generated
            generated_texts = generated_texts[:k]
        # 생성된 응답 출력
        for text in generated_texts:
            unique_texts.append({"situation": keyword, "response": text})
            total_generated += 1
            print(f"Generated text for '{keyword}': {text}")
    # 생성된 고유 응답 반환
    return unique_texts


# 데이터 생성 실행
gpt_texts = generate_data(traffic_keywords, gold_samples)

# 생성된 데이터 JSON 파일로 저장
with open("generated_traffic_data.json", "w", encoding="utf-8") as f:
    json.dump(gpt_texts, f, ensure_ascii=False, indent=4)

print("\n데이터셋이 'generated_traffic_data.json' 파일로 저장되었습니다.")
