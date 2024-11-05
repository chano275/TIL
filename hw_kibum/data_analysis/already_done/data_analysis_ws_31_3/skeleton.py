# 필요한 패키지 설치
# TODO: 패키지를 설치하고 아래의 평가예시 모델을 테스트할 수 있도록 준비해주세요.
# !pip install pandas mlflow openai tiktoken tenacity evaluate textstat torch transformers

# 1. 교통 데이터 처리를 위한 LLM 성능 분석 환경 설정
# OpenAI와 MLflow를 사용하여 교통 관련 질문에 대해 LLM 성능을 평가할 준비를 합니다.

import openai
import pandas as pd
import mlflow
import os

# OpenAI API 키 설정
# API 키를 환경 변수로 설정하여 OpenAI 모델에 접근할 수 있도록 합니다.
# 참고: https://platform.openai.com/docs/quickstart/authentication
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# 교통 관련 질문 및 예상 답변 데이터 생성
# 평가를 위해 모델에 제공할 교통 관련 질문(inputs)과 그에 대한 기준 답변(ground_truth)을 정의합니다.
eval_df = pd.DataFrame({
    "inputs": [
        "현재 시내 교통 상황은 어떤가요?",
        "출퇴근 시간의 혼잡을 피하려면 어떻게 해야 하나요?",
        "교통 혼잡의 원인은 무엇인가요?",
        "교통 체증이 공기 질에 미치는 영향은 무엇인가요?",
    ],
    "ground_truth": [
        "현재 시내 교통은 대부분 지역에서 차량 속도가 느려 혼잡한 상태입니다.",
        "출퇴근 혼잡을 피하려면 피크 시간을 피해서 일찍 또는 늦게 이동하는 것이 좋습니다.",
        "교통 혼잡은 주로 사고, 도로 공사, 높은 차량량 등에 의해 발생합니다.",
        "교통 체증은 차량 배출가스 증가로 인해 공기 질이 악화될 수 있습니다.",
    ],
})

#########################################################################################################
"""
MLflow:
MLflow는 머신러닝 실험을 관리하고 추적할 수 있는 오픈 소스 플랫폼입니다. 
모델 성능 평가와 버전 관리, 배포를 한 곳에서 처리할 수 있어 모델 개발과 운영을 효율적으로 관리할 수 있습니다.

### MLflow의 주요 기능:
1. 실험 관리 및 추적: 모델 성능, 하이퍼파라미터, 메트릭 등을 기록하고, 실험 결과를 비교할 수 있습니다.
2. 모델 버전 관리: 학습된 모델을 버전별로 관리하고, 필요한 경우 이전 버전으로 쉽게 복구할 수 있습니다.
3. 자동 메트릭 로깅: 기본 및 사용자 정의 메트릭을 기록하여 모델 평가를 지원합니다.
4. 모델 배포: 모델을 REST API로 배포하거나 컨테이너 환경에서 관리할 수 있습니다.
5. 시각화 및 분석: 웹 UI를 통해 실험 결과를 시각적으로 분석하고, 실험 간 성능 차이를 쉽게 비교할 수 있습니다.

### 활용 분야:
- 머신러닝 모델 실험 및 성능 분석
- 모델의 자동화된 배포 및 운영 관리
- 재현 가능한 머신러닝 연구 환경 구축

이번 코드에서는 모델 성능 평가와 메트릭 로깅을 MLflow로 설정하여, 교통 데이터 질문에 대한 LLM의 성능을 분석합니다.
"""
# 2. LLM 성능 분석을 위한 프레임워크 준비
# MLflow를 사용하여 OpenAI 모델을 설정하고, 모델 평가를 위한 기본 환경을 구축합니다.

# 동일한 세션 내에서 여러 평가를 수행하기 위해 하나의 mlflow.start_run() 내부에서 모든 평가 작업을 진행합니다.
with mlflow.start_run() as run:
    # 참고: mlflow.start_run() 함수는 새로운 MLflow 실험 실행을 시작합니다.
    # MLflow가 해당 실행에 대한 로그와 결과를 저장합니다.
    # 참고 페이지: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run
    # - `run_name`: 실행의 이름을 설정합니다 (선택적).
    # - `nested`: 실행이 중첩된 실행인 경우 True로 설정합니다.
    # - `tags`: 실행에 추가할 메타데이터 태그를 지정하는 딕셔너리입니다.

    # 시스템 프롬프트 설정
    # 모델에 교통 관련 질문에 한 문장으로 답변하도록 지시하는 시스템 프롬프트를 설정합니다.
    system_prompt = "다음 교통 관련 질문에 한 문장으로 답변하세요."
    
    # 모델 설정 및 MLflow에 로깅
    # gpt-4o-mini 모델을 사용하여 질문에 응답하는 모델을 설정하고, MLflow에 로깅합니다.
    # 참고 페이지: https://mlflow.org/docs/latest/python_api/openai/index.html
    # - `model`: 사용 모델 이름을 지정합니다.
    # - `task`: 수행할 작업 유형을 설정합니다 (예: completion).
    # - `artifact_path`: MLflow에 저장될 모델 아티팩트의 경로입니다.
    # - `messages`: 모델과의 메시지 대화 흐름을 설정합니다 (시스템 메시지, 사용자 메시지 등).
    # TODO: 아래의 빈칸에 해당 함수와 함수에 들어가야 할 인자를 채워주세요!
    traffic_qa_model = __________________(
        model="gpt-4o-mini",  # 파라미터로 지정된 모델 이름, 예: gpt-4o-mini
        ___________=openai.chat.completions,  # OpenAI 작업 유형을 지정, 예: chat completions
        ___________="model",  # MLflow에 저장될 모델의 경로 지정
        ___________=[
            {"role": "system", "content": system_prompt},  # 시스템 역할로 모델 지시문 설정
            {"role": "user", "content": "{question}"},  # 사용자 입력 역할 설정
        ],
    )
    
#########################################################################################################

# 3. LLM 성능 분석을 위한 사용자 정의 메트릭 설정
# 답변 유사도와 응답 전문성을 평가하기 위한 사용자 정의 메트릭을 설정합니다.

    # 참고 페이지: https://mlflow.org/docs/latest/python_api/openai/index.html
    # - `name`: 생성할 메트릭의 이름을 지정합니다.
    # - `definition`: 메트릭의 목적 및 정의를 설명합니다.
    # - `grading_prompt`: 평가 기준 및 점수 척도에 대해 설명하는 평가 프롬프트입니다.
    # - `examples`: 평가 예시를 담은 리스트입니다 (각 예시는 평가 항목 및 점수를 포함합니다).
    # - `model`: 평가에 사용할 모델을 지정합니다.
    # - `version`: 메트릭의 버전을 설정합니다.
    # - `parameters`: 모델의 온도 등 추가 파라미터를 딕셔너리 형태로 지정합니다.

    from mlflow.metrics.genai import EvaluationExample, answer_similarity, make_genai_metric

    # 답변 유사도 평가 메트릭 정의
    # 모델 응답이 기준 예시와 얼마나 유사한지를 평가하기 위한 예시 설정입니다.
    example = EvaluationExample(
        input="교통 혼잡의 원인은 무엇인가요?",  # 모델에게 주어지는 입력 질문
        output="교통 혼잡은 주로 높은 차량량이나 도로 차단으로 인해 발생합니다.",  # 모델의 이상적인 응답
        score=4,  # 응답이 기준에 얼마나 부합하는지에 대한 점수 (최대 4점)
        justification="답변이 정확하며 일반적인 교통 혼잡의 원인과 잘 일치합니다."  # 점수에 대한 근거 설명
    )

    # answer_similarity 함수로 유사도 메트릭 정의
    # 정의된 예시 데이터를 사용하여 모델 응답의 유사도를 평가하는 메트릭을 생성합니다.
    # 참고: https://mlflow.org/docs/latest/python_api/mlflow.metrics.html#mlflow.metrics.genai.answer_similarity
    # TODO: 유사도 비교를 위한 mlflow 함수를 사용해서 구현해주세요.
    # model은 openai의 gpt-4를 이용하세요.
    
    answer_similarity_metric = ______________(
        model=______________,  # 비교할 OpenAI 모델 지정
        examples=[example]  # 유사도 평가 예시 리스트
    )

    # 응답 전문성 평가 메트릭 정의
    # 참고: https://mlflow.org/docs/latest/python_api/mlflow.metrics.html#mlflow.metrics.genai.make_genai_metric
    # TODO: gen ai의 메트릭 설정을 위해 위의 페이지를 참고하여 빈칸을 완성해주세요.
    # 파라미터는 해당 documenation을 통해 확인할 수 있습니다.
    professionalism_metric = ______________(
        name="professionalism",
        ______________="전문성은 격식 있고 적절한 언어 사용을 통해 문맥에 맞는 응답을 측정합니다.",
        ______________=(
            "전문성: 응답이 격식 있는 톤으로 작성되었는지 여부를 아래 기준에 따라 점수로 매기세요...\n"
            "score: 점수 (숫자)\n"
            "justification: 평가 근거"
        ),
        examples=example,
        model=______________,  # 비교할 OpenAI 모델 지정
        version="v1",
        parameters={_____________}, # temperature를 0으로 설정하세요.
    )
    print("답변 유사도 및 전문성 평가를 위한 메트릭이 설정되었습니다.")
    print(f"professionalism_metric: {professionalism_metric}")
    print("\n")