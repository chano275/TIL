# 필요한 패키지 설치
# !pip install pandas mlflow openai nltk sklearn

# 1. 금융 데이터 처리를 위한 LLM 응용 분석 환경 설정
# OpenAI와 MLflow를 사용하여 금융 관련 질문에 대해 LLM 성능을 평가할 준비를 합니다.

import os
import pandas as pd
import mlflow
import openai
import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mlflow.metrics.genai import EvaluationExample, make_genai_metric

nltk.download('wordnet')
# nltk 데이터 다운로드 (한국어 WordNet 사용을 위해 필요)
# 참고 링크: https://www.nltk.org/nltk_data/
nltk.download('punkt')
nltk.download('omw-1.4')  # 한국어 WordNet 사용을 위해 필요

# OpenAI API 키 설정
# 환경 변수에서 API 키를 불러와 OpenAI 모델에 접근할 수 있도록 합니다.
# OpenAI API 키 설정 (환경 변수 활용) 참고: https://platform.openai.com/docs/quickstart/authentication
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = os.getenv('')

# 2. 금융 관련 질문 데이터셋 준비
# 금융 관련 질문(inputs)과 그에 대한 기준 답변(ground_truth)을 정의합니다.
finance_df = pd.DataFrame({
    "inputs": [
        "주식 시장에서 분산 투자의 의미는 무엇인가요?",
        "채권 투자로 얻을 수 있는 주요 이점은 무엇인가요?",
        "ETF는 무엇의 약자인가요?",
        "인플레이션이 경제에 미치는 영향은 무엇인가요?"
    ],
    "ground_truth": [
        "분산 투자는 위험을 줄이기 위해 다양한 자산에 투자하는 것입니다.",
        "채권 투자는 안정적인 이자 수익을 제공합니다.",
        "ETF는 상장지수펀드의 약자입니다.",
        "인플레이션은 화폐 가치의 감소로 구매력을 하락시킵니다."
    ],
})

# 3. 데이터 증강 함수 구현
# 주어진 문장에서 단어를 동의어로 치환하여 새로운 문장을 생성합니다.
# 참고: https://www.nltk.org/howto/wordnet.html
def synonym_replacement(sentence):
    # 문장을 단어로 토큰화
    # nltk.word_tokenize() 참고: https://www.nltk.org/book/ch03.html
    words = nltk.word_tokenize(sentence)
    new_words = []
    # 각 단어에 대해 동의어를 찾아 치환
    for word in words:
        synonyms = wordnet.synsets(word, lang='eng')  # 영어 단어의 동의어를 찾습니다
        if synonyms:
            # lemma 참고: https://www.nltk.org/howto/wordnet.html#lemmas
            # 첫 번째 동의어 집합의 첫번째 레마(단어의 기본형)를 선택합니다.
            lemma = synonyms[0].lemmas()[0].name() # 첫 번째 동의어의 첫 번째 레마를 사용
            # 새로운 문장에 동의어를 추가합니다.
            new_words.append(lemma)
        else:
            new_words.append(word)
    return ' '.join(new_words)

# 증강된 데이터를 새로운 컬럼에 추가
finance_df['augmented_inputs'] = finance_df['inputs'].apply(synonym_replacement)

# 4. 데이터 필터링 함수 구현
# 원본 문장과 증강된 문장의 유사도를 계산하여 품질이 낮은 데이터를 필터링합니다.
# 참고: TfidfVectorizer는 텍스트를 벡터로 변환하고, cosine_similarity는 두 벡터의 유사도를 계산합니다.
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
# 두 문장의 유사도를 비교하여 특정 값 이하인 경우 필터링하는 조건을 빈칸에 완성하세요.
def filter_low_quality(original, augmented):
    # TfidVectorizer 참고: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    vectorizer = TfidfVectorizer() # TF-IDF 벡터라이저 객체 생성

    # TODO: 벡터라이저로 두 문장을 벡터로 변환하여 'vectors'에 할당합니다.
    # vectorizer.fit_transform()를 사용하여 두 문장을 벡터로 변환합니다.
    # 참고: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    # 벡터라이저로 original과 augmented를 벡터화합니다.
    vectors = vectorizer.fit_transform([original, augmented]) # 벡터라이저로 original과 augmented를 벡터화합니다.

    # TODO: 두 벡터의 유사도를 계산하여 'similarity'에 할당합니다.
    # cosine_similarity()를 사용하여 두 벡터의 유사도를 계산합니다.
    # 유사도를 측정할 두 벡터는 vectors[0:1]과 vectors[1:2]입니다.
    # 참고: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])  # 두 벡터 간 유사도 계산

    return similarity[0][0] > 0.7  # 유사도가 0.7 이하인 경우 필터링

# 필터링 결과를 새로운 컬럼에 추가
finance_df['is_high_quality'] = finance_df.apply(
    lambda row: filter_low_quality(row['inputs'], row['augmented_inputs']), axis=1
)

# 품질이 높은 데이터만 선택
filtered_df = finance_df[finance_df['is_high_quality']]

# 5. LLM 성능 평가를 위한 프레임워크 준비
# MLflow를 사용하여 평가를 수행할 준비를 합니다.
with mlflow.start_run() as run:
    # 시스템 프롬프트 설정
    # 모델에게 금융 관련 질문에 한 문장으로 답변하도록 지시하는 시스템 프롬프트를 설정합니다.
    system_prompt = "다음 금융 관련 질문에 한 문장으로 답변하세요."

    # 모델 설정 및 MLflow에 로깅
    finance_qa_model = mlflow.openai.log_model(
        # gpt-4o-mini 모델을 사용하여 질문에 응답하는 모델을 설정하고, MLflow에 로깅합니다.
        # 참고 링크: https://mlflow.org/docs/latest/python_api/openai.html#mlflow.openai.log_model
        model="gpt-4o-mini",  # 사용할 모델 이름 지정
        task='chat.completions',  # 수행할 작업 유형 설정
        artifact_path="model",  # MLflow에 저장될 모델의 경로 지정
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "{inputs}"}
        ],
    )

    # 사용자 정의 메트릭 설정
    # 기준 답변과 모델 답변의 유사성을 평가하기 위한 메트릭을 설정합니다.
    example_similarity = EvaluationExample(
        input="주식 시장에서 분산 투자의 의미는 무엇인가요?",
        output="분산 투자는 위험을 줄이기 위해 다양한 자산에 투자하는 것입니다.",
        score=5,
        justification="답변이 분산 투자의 개념을 정확하고 완벽하게 설명하고 있습니다."
    )

    answer_similarity_metric = make_genai_metric(
        name="answer_similarity",  # 메트릭의 이름 지정
        definition="모델의 응답이 기준 답변과 얼마나 의미적으로 유사한지를 측정합니다.",  # 메트릭의 목적 및 정의 설명
        grading_prompt=(
            "다음 기준에 따라 모델의 응답이 기준 답변과 얼마나 의미적으로 유사한지 평가하고, 점수와 평가 근거를 작성하세요.\n"
            "score: (0-5)\n"
            "justification: 설명\n"
            "평가 기준: 0점은 전혀 유사하지 않음을 의미하며, 5점은 완벽하게 유사함을 의미합니다."
        ),  # 평가 기준 및 점수 척도 설명
        examples=[example_similarity],
        model="openai:/gpt-4o-mini",  # 평가에 사용할 모델 지정
        version="v1",  # 메트릭의 버전 지정
        parameters={"temperature": 0},
    )

    # 6. 모델 평가 실행
    # MLflow의 evaluate 메서드를 사용하여 모델을 평가합니다.
    # 참고 링크: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
    data = filtered_df[['augmented_inputs', 'ground_truth']].copy()
    data = data.rename(columns={'augmented_inputs': 'inputs'})

    results = mlflow.evaluate(
        model=finance_qa_model.model_uri,  # 평가할 모델의 URI를 지정
        data=data,  # 평가에 사용할 데이터프레임 지정
        targets="ground_truth",  # 기준 답변 컬럼명 지정
        model_type="question-answering",  # 모델 유형 지정
        evaluators="default",  # 평가 지표 지정
        extra_metrics=[answer_similarity_metric],  # 사용자 정의 메트릭 추가
    )

    # 7. 평가 결과 출력
    print("모델 평가가 완료되었습니다.")
    print("평가 메트릭:")
    print(results.metrics)
    print("\n")

    # 8. 각 질문에 대한 모델 응답 및 평가 결과 출력
    eval_results = results.artifacts["eval_results_table"].content["data"]
    columns = results.artifacts["eval_results_table"].content["columns"]
    eval_df = pd.DataFrame(eval_results, columns=columns)

    for i, row in eval_df.iterrows():
        print(f"질문: {row['inputs']}")
        print(f"모델 응답: {row['outputs']}")
        print(f"기대 답변: {row['ground_truth']}")
        print(f"Answer Similarity Score: {row['answer_similarity/v1/score']}")
        print(f"Answer Similarity Justification: {row['answer_similarity/v1/justification']}")
        print("\n")
