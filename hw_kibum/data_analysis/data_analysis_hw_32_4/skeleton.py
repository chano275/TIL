# 과제 2: AI 데이터 검수

# 필요한 라이브러리 임포트
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# OpenAI API Key 설정 - API 키를 환경 변수에 설정하여 OpenAI 서비스를 사용합니다.
os.environ["OPENAI_API_KEY"] = ""

# 임베딩 모델 설정 - 텍스트 데이터를 수치 벡터로 변환하기 위한 임베딩 모델
# OpenAIEmbeddings() 참고: https://python.langchain.com/docs/integrations/text_embedding/openai/#instantiation
# 이 문제에선, OpenAIEmbeddings()를 사용하여 임베딩 모델을 설정합니다.
# TODO: 임베딩 모델을 설정하기 위한 함수를 완성하세요.
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# 생성된 데이터 JSON 파일 불러오기
with open("./data/generated_traffic_data.json", "r", encoding="utf-8") as f:
    gpt_texts = json.load(f)


# 유사도 필터링 함수 정의
# 이 문제에선 생성된 응답 리스트에서 유사한 응답을 제거하여 고유한 응답만 필터링합니다.
def filter_similar_responses(gpt_texts, similarity_threshold=0.9):
    """
        유사도가 높은 응답을 제거하여 고유한 응답만 필터링합니다.

        Parameters:
        - gpt_texts (list): 생성된 응답 리스트
        - similarity_threshold (float): 응답 간 유사도를 비교할 임계값

        Returns:
        - final_unique_texts (list): 유사도 기준으로 필터링된 고유 응답 리스트
    """
    final_unique_texts = []
    final_unique_embeddings = []

    # 생성된 응답 리스트를 순회하며 유사도가 높은 응답을 제거합니다.
    for text in gpt_texts:
        response = text["response"]
        emb = embedding_model.embed_query(response)

        # 유사도 검사
        if final_unique_embeddings:
            # consine_similarity() 함수를 사용하여 두 벡터 간의 코사인 유사도를 계산합니다.
            # cosine_similarity() 참고: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
            # TODO: 두 벡터 간의 코사인 유사도를 계산하는 코드를 작성하세요.
            sims = cosine_similarity([emb], final_unique_embeddings)[0]
            if max(sims) > similarity_threshold:
                print(f"제외된 유사한 응답: {response} (유사도 {max(sims):.2f})")
                continue

        # 고유한 응답으로 간주하여 리스트에 추가
        final_unique_embeddings.append(emb)
        final_unique_texts.append(text)

    return final_unique_texts


# 유사도 필터링 실행
filtered_texts = filter_similar_responses(gpt_texts)

# 최종 데이터셋을 상황별로 정리하여 JSON 파일로 저장
output_data = {}
# 상황별로 응답을 그룹화하여 최종 데이터셋을 생성합니다.
for item in filtered_texts:
    situation = item["situation"]
    response = item["response"]
    if situation not in output_data:
        output_data[situation] = []
    output_data[situation].append(response)

print("\n검수된 최종 교통 데이터셋:")

# 상황별로 응답을 출력합니다.
for situation, responses in output_data.items():
    print(f"{situation}:")
    # 각 상황에 대한 응답을 출력합니다.
    for response in responses:
        print(f" - {response}")

# 검수된 데이터셋 JSON 파일로 저장
with open("final_traffic_data.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print("\n데이터셋이 'final_traffic_data.json' 파일로 저장되었습니다.")
