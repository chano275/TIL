# 필요한 패키지 설치
# TODO: 패키지를 설치하고 아래의 평가예시 모델을 테스트할 수 있도록 준비해주세요.
# !pip install pandas mlflow openai tiktoken tenacity evaluate textstat torch transformers

# 1. 교통 데이터 처리를 위한 LLM 성능 분석 환경 설정
# OpenAI와 MLflow를 사용하여 교통 관련 질문에 대해 LLM 성능을 평가할 준비를 합니다.

import pandas as pd

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
print("교통 관련 질문에 대한 평가 데이터셋이 생성되었습니다.")
print(f"eval_df: {eval_df}")
print("\n")
