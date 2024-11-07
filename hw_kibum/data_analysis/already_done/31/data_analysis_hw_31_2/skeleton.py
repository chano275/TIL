# 필요한 패키지 설치
# !pip install pandas mlflow openai tiktoken tenacity evaluate textstat torch transformers

# 1. 금융 데이터 처리를 위한 LLM 성능 평가 환경 설정
# OpenAI와 MLflow를 사용하여 금융 관련 질문에 대해 LLM 성능을 평가할 준비를 합니다.

import os
import pandas as pd
import mlflow
import openai
from mlflow.metrics.genai import EvaluationExample, make_genai_metric



# OpenAI API 키 설정
# 환경 변수에서 API 키를 불러와 OpenAI 모델에 접근할 수 있도록 합니다.
# OpenAI API 키 설정 (환경 변수 활용) 참고: https://platform.openai.com/docs/quickstart/authentication
# os.getenv() 참고: https://docs.python.org/3/library/os.html#os.environ
openai.api_key = os.getenv('')
os.environ['OPENAI_API_KEY'] = ''


# pandas 출력 설정 변경 (필요에 따라 설정)
# 데이터프레임 출력 시 모든 열이 표시되도록 설정합니다.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# 금융 관련 질문 및 기대 답변 데이터셋 준비
# 평가를 위해 모델에 제공할 금융 관련 질문(inputs)과 그에 대한 기준 답변(ground_truth)을 정의합니다.
finance_df = pd.DataFrame({
    "inputs": [
        "주식 시장에서 분산 투자의 의미는 무엇인가요?",
        "채권 투자의 주요 이점은 무엇인가요?",
        "ETF는 무엇의 약자인가요?",
        "인플레이션이 경제에 미치는 영향은 무엇인가요?"
    ],
    "ground_truth": [
        "분산 투자는 위험을 줄이기 위해 다양한 자산에 투자하는 것입니다.",
        "채권 투자는 안정적인 이자 수익을 제공합니다.",
        "ETF는 상장지수펀드의 약자입니다.",
        "인플레이션은 화폐의 가치를 감소시켜 구매력을 하락시킵니다."
    ],
})

#########################################################################################################

# 2. LLM 성능 평가를 위한 프레임워크 준비
# MLflow를 사용하여 OpenAI 모델을 설정하고, 모델 평가를 위한 기본 환경을 구축합니다.

# 동일한 세션 내에서 여러 평가를 수행하기 위해 하나의 mlflow.start_run() 내부에서 모든 평가 작업을 진행합니다.
with mlflow.start_run() as run:
    # 시스템 프롬프트 설정
    # 모델에게 금융 관련 질문에 한 문장으로 답변하도록 지시하는 시스템 프롬프트를 설정합니다.
    system_prompt = "다음 금융 관련 질문에 한 문장으로 답변하세요."

    # 모델 설정 및 MLflow에 로깅
    # gpt-4o-mini 모델을 사용하여 질문에 응답하는 모델을 설정하고, MLflow에 로깅합니다.
    # 참고 페이지: https://mlflow.org/docs/latest/python_api/openai/index.html#mlflow.openai.log_model
    # - 'model': 사용할 모델 이름을 지정합니다.이 과제에선 gpt-4o-mini 모델을 사용합니다.
    # - 'task': 수행할 작업 유형을 설정합니다. 이 과제에선 'chat.completions' 작업을 수행합니다.
    # - 'artifact_path': MLflow에 저장될 모델 아티팩트의 경로입니다. 이 과제에선 'model' 경로에 저장합니다.
    # - 'messages': 모델과의 메시지 대화 흐름을 설정합니다 (시스템 메시지, 사용자 메시지 등). 이 과제에선 시스템 프롬프트와 사용자 질문을 설정합니다.
    # TODO: 아래의 빈칸에 해당 함수와 함수에 들어가야 할 인자를 채워주세요!
    finance_qa_model = mlflow.openai.log_model(
        model="gpt-4o-mini",  # 사용할 모델 이름 지정
        task='chat.completions',  # 수행할 작업 유형 설정
        artifact_path="model",  # MLflow에 저장될 모델의 경로 지정
        messages=[
            {"role": "system", "content": system_prompt},  # 시스템 프롬프트 설정
            {"role": "user", "content": "{question}"}  # 사용자 입력 설정
        ],
    )

    #########################################################################################################

    # 3. LLM 성능 평가를 위한 사용자 정의 메트릭 설정
    # 답변 유사도와 응답 전문성을 평가하기 위한 사용자 정의 메트릭을 설정합니다.

    # 답변 유사도 평가 메트릭 정의
    # 모델 응답이 기준 답변과 얼마나 의미적으로 유사한지를 평가하기 위한 예시를 설정합니다.
    example_similarity = EvaluationExample(
        input="주식 시장에서 분산 투자의 의미는 무엇인가요?",  # 모델에게 주어지는 입력 질문
        output="분산 투자는 위험을 줄이기 위해 다양한 자산에 투자하는 것입니다.",  # 모델의 이상적인 응답
        score=5,  # 응답이 기준에 얼마나 부합하는지에 대한 점수 (0-5점)
        justification="답변이 정확하며 분산 투자의 개념을 완벽하게 설명하고 있습니다."  # 점수에 대한 근거 설명
    )

    # answer_similarity_metric 정의
    # make_genai_metric 함수 참고: https://mlflow.org/docs/latest/python_api/mlflow.metrics.html#mlflow.metrics.genai.make_genai_metric
    # make_genai_metric 함수를 사용하여 응답 유사성 평가 메트릭을 생성합니다.
    # name: 생성할 메트릭의 이름을 지정합니다. 이 과제에선 "answer_similarity"로 설정합니다.
    # definition: 메트릭의 목적 및 정의를 설명합니다.
    # grading_prompt: 평가 기준 및 점수 척도에 대해 설명하는 평가 프롬프트입니다.
    # examples: 평가 예시를 담은 리스트입니다 (각 예시는 평가 항목 및 점수를 포함합니다).
    # model: 평가에 사용할 모델을 지정합니다. 이 과제에선 "openai:/gpt-4o-mini" 모델을 사용합니다.
    # version: 메트릭의 버전을 설정합니다. 이 과제에선 "v1"로 설정합니다.
    # parameters: 모델의 온도 등 추가 파라미터를 딕셔너리 형태로 지정합니다.
    # TODO: 아래의 빈칸에 해당 함수의 인자를 채워주세요
    answer_similarity_metric = make_genai_metric(
        name="answer_similarity",
        definition="응답 유사성은 모델의 응답이 기준 답변과 얼마나 의미적으로 유사한지를 측정합니다.",
        grading_prompt=(
            "다음 기준에 따라 모델의 응답이 기준 답변과 얼마나 의미적으로 유사한지 평가하고, 점수와 평가 근거를 한국어로 작성하세요.\n"
            "score: 점수 (0-5)\n"
            "justification: 평가 근거\n"
            "평가 기준: 0점은 전혀 유사하지 않음을 의미하며, 5점은 완벽하게 유사함을 의미합니다."
        ),
        examples=[example_similarity],
        model="openai:/gpt-4o-mini",
        version="v1",
        parameters={"temperature": 0},
    )

    # 응답 전문성 평가 메트릭 정의
    # 모델의 응답이 얼마나 전문적인지를 평가하기 위한 예시를 설정합니다.
    professionalism_example = EvaluationExample(
        input="ETF는 무엇의 약자인가요?",
        output="ETF는 상장지수펀드의 약자입니다.",
        score=4,
        justification="응답이 격식 있고 정확한 정보를 제공합니다."
    )

    # professionalism_metric 정의
    # make_genai_metric 함수를 사용하여 응답 전문성 평가 메트릭을 생성합니다.
    # name: 생성할 메트릭의 이름을 지정합니다. 이 과제에선 "professionalism"으로 설정합니다.
    # definition: 메트릭의 목적 및 정의를 설명합니다.
    # grading_prompt: 평가 기준 및 점수 척도에 대해 설명하는 평가 프롬프트입니다.
    # examples: 평가 예시를 담은 리스트입니다 (각 예시는 평가 항목 및 점수를 포함합니다).
    # model: 평가에 사용할 모델을 지정합니다. 이 과제에선 "openai:/gpt-4o-mini" 모델을 사용합니다.
    # version: 메트릭의 버전을 설정합니다. 이 과제에선 "v1"로 설정합니다.
    # parameters: 모델의 온도 등 추가 파라미터를 딕셔너리 형태로 지정합니다.
    professionalism_metric = make_genai_metric(
        name="professionalism",
        definition="전문성은 적절한 언어 사용과 정확한 정보를 통해 응답의 품질을 측정합니다.",
        grading_prompt=(
            "전문성: 응답의 정확성, 명확성, 그리고 공손한 언어 사용을 기반으로 평가합니다. "
            "높은 점수는 정확하고 신뢰할 수 있는 정보를 명확하고 공손하게 전달하는 응답에 부여됩니다.\n"
        
            "score: 점수 (0-4)\n"
            "justification: 평가 근거"
        ),
        examples=[professionalism_example],
        model="openai:/gpt-4o-mini",
        version="v1",
        parameters={"temperature": 0},
    )

    #########################################################################################################

    # 4. LLM의 성능 평가 실행
    # MLflow를 사용하여 모델을 평가하고, 결과를 출력합니다.

    # 모델 평가 실행
    # 참고 페이지: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
    # - 'model_uri': 평가할 모델의 URI입니다. 이 과제에선 finance_qa_model.model_uri를 사용합니다.
    # - 'data': 모델에 제공할 입력 데이터 또는 예시입니다.
    # - 'targets': 예측 결과와 비교할 기준 정답 데이터입니다. 이 과제에선 정의된 df에서의 정답의 key인 "ground_truth"로 설정합니다.
    # - 'model_type': 평가할 모델의 유형입니다. 이 과제에선 "question-answering"으로 설정합니다.
    #   model type은 'classifier', 'regressor', 'question-answering', 'text-summarization', 'text', 'retriever' 이 가능합니다
    # - 'evaluators': 사용할 평가 지표입니다. 이 과제에선 "default"로 설정합니다.
    #   참고: https://mlflow.org/docs/latest/model-evaluation/index.html
    # - 'extra_metrics': 사용자가 정의한 metric을 의미합니다. 조금 전 정의한 두 metric을 사용합니다.
    # TODO: 아래의 빈칸에 필요한 값을 넣어주세요.
    results = mlflow.evaluate(
        model=finance_qa_model.model_uri,  # 모델 URI를 사용
        data=finance_df,
        targets="ground_truth",
        model_type="question-answering",
        evaluators=["default"],  # evaluators는 리스트 형태로 입력
        extra_metrics=[answer_similarity_metric, professionalism_metric],
    )

    # 평가 결과 출력
    print("모델 평가가 완료되었습니다.")
    print("평가 메트릭:")
    print(results.metrics)
    print("\n")

    #########################################################################################################

    # 5. 각 질문에 대한 모델 응답 및 평가 결과 출력
    # 평가 결과를 데이터프레임으로 변환하여 각 질문에 대한 모델의 응답과 평가 점수를 확인합니다.

    # 평가 결과 데이터 추출
    eval_results = results.artifacts["eval_results_table"].content["data"]
    columns = results.artifacts["eval_results_table"].content["columns"]

    # eval_results를 데이터프레임으로 변환
    eval_df = pd.DataFrame(eval_results, columns=columns)

    # 컬럼 이름 출력
    print("Available columns:", eval_df.columns)

    # 각 질문에 대한 응답 및 평가 결과 출력
    for i, row in eval_df.iterrows():
        print(f"질문: {row['inputs']}")
        print(f"모델 응답: {row['outputs']}")
        print(f"기대 답변: {row['ground_truth']}")
        print(f"answer_similarity score: {row['answer_similarity/v1/score']}")
        print(f"answer_similarity justification: {row['answer_similarity/v1/justification']}")
        print(f"professionalism score: {row['professionalism/v1/score']}")
        print(f"professionalism justification: {row['professionalism/v1/justification']}")
        print("\n")
