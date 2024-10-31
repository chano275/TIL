import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model_and_tokenizer(model_name):  # 모델과 토크나이저를 로드하는 함수
    """
    사전 훈련된 LLM 모델과 토크나이저를 로드합니다.
    사전 훈련된 모델(LLM, Pre-trained Language Model)은 이미 대량의 텍스트 데이터를 사용해 학습된 모델로,
    특정 작업에 맞게 추가 학습을 할 수 있습니다.
    """
    # AutoModelForCausalLM과 AutoTokenizer를 사용해 모델과 토크나이저를 로드
    # AutoModelForCausalLM 참고: https://huggingface.co/transformers/model_doc/gpt2.html#aut
    # AutoTokenizer 참고: https://huggingface.co/transformers/model_doc/gpt2.html#autotokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 패딩 토큰 설정: 모델이 입력 텍스트 길이를 일정하게 맞출 수 있도록 패딩 토큰을 사용 / 패딩 토큰이 없는 경우, 'eos_token'(문장의 끝 토큰)을 패딩 토큰으로 사용하여 길이를 맞춥니다.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰을 eos_token으로 설정

    return model, tokenizer

def prepare_model_for_instruction_tuning(model):  # Instruction tuning을 위해 모델을 준비하는 함수
    """
    모델을 학습 모드로 설정하고, 특정 파라미터만 학습 가능하도록 설정합니다.
    미세 조정(Instruction Tuning)은 모델의 일부 파라미터만 업데이트하여 효율적인 학습을 수행하는 방법입니다.
    여기서는 마지막 레이어의 파라미터만 학습 가능하게 설정하여, 모델의 성능을 특정 작업에 맞게 개선할 수 있습니다.
    """
    for param in model.parameters():    # 모든 파라미터를 학습 불가능하게 설정: 기존의 사전 훈련된 가중치를 고정
        param.requires_grad = False

    # TODO: 마지막 레이어의 파라미터만 학습 가능하도록 설정하세요
    """
    model.transformer.h는 모델의 모든 레이어를 포함하는 리스트이며, h[-1]은 마지막 레이어를 의미
    model.transformer.h[-1]을 사용하여 마지막 레이어를 가져오고, 해당 레이어의 파라미터들의 requires_grad를 True로 설정합니다.
    
    해당 레이어에 존재하는 모든 파라미터를 가져오기 위해서는 parameters() 함수를 사용합니다.
    마지막 레이어의 모든 파라미터에 대해 requires_grad를 True로 설정하면 해당 레이어만 학습됩니다.    
    """
    # 참고:
    # - nn.Module의 method인 parameters:
    #   https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.parameters
    # - PyTorch에서 파라미터의 requires_grad 사용법:
    #   https://pytorch.org/docs/stable/autograd.html#torch.Tensor.requires_grad
    # - GPT-2 모델의 구조에 대한 설명:
    #   https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Model
    for param in model.transformer.h[-1].parameters():
        param.requires_grad = True

    model.train()    # 모델을 학습 모드로 전환해서 학습할 준비를 합니다.
    if torch.cuda.is_available():        # GPU가 사용 가능한 경우, 모델을 GPU로 이동하여 학습 속도를 높입니다.
        model = model.to('cuda')
        print("모델이 GPU로 이동되었습니다.")
    else:        # GPU를 사용할 수 없는 경우, CPU에서 학습하게 됩니다.
        print("GPU를 사용할 수 없습니다. CPU에서 학습합니다.")

    # 학습 가능한 파라미터 수를 출력: 얼마나 많은 파라미터가 업데이트될 수 있는지 확인
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Instruction tuning을 위한 학습 가능한 파라미터 수: {num_trainable_params}")

    return model


def load_and_tokenize_data(tokenizer, file_path, max_length=512):  # 데이터를 로드하고 토큰화하는 함수
    """
    JSON 파일에서 데이터를 불러와서 토크나이저를 사용해 토큰화합니다.
    텍스트 데이터를 모델이 이해할 수 있는 숫자 형태로 변환하는 과정이며, 이를 통해 모델이 학습할 수 있는 형식으로 데이터를 준비합니다.
    max_length 옵션을 사용해 텍스트 길이를 일정하게 맞추고, 필요 시 잘라내거나 패딩을 추가합니다.
    """
    # TODO: JSON 파일에서 데이터를 불러오고 각 텍스트를 토큰화하세요
    # 1. JSON 파일을 열어 데이터를 로드합니다.
    #    - open() 함수를 사용하여 파일을 읽기 모드로 열고, json.load()를 사용하여 데이터를 파싱합니다.
    #    - 파일 경로는 file_path 변수에 저장되어 있습니다.

    # 2. 데이터에서 'chunks' 키를 사용하여 텍스트 리스트를 가져옵니다.
    #    - 데이터는 {"chunks": [...]} 형태로 저장되어 있으므로, data["chunks"]로 접근합니다.

    # 3. 각 텍스트 데이터를 토큰화합니다.
    #    - tokenizer.encode()를 사용하여 텍스트를 토큰 시퀀스로 변환합니다.
    #    - max_length, truncation, padding 등의 옵션을 설정하여 시퀀스 길이를 맞춥니다.
    # 참고:
    # - json 모듈의 사용법: https://docs.python.org/ko/3/library/json.html
    # - 파일 입출력 관련 정보: https://docs.python.org/ko/3/tutorial/inputoutput.html#reading-and-writing-files
    # - Hugging Face Tokenizer의 encode 함수: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.encode
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)['chunks']

    # 각 텍스트 데이터를 토큰화: 텍스트를 숫자로 변환하여 모델이 처리할 수 있도록 함
    tokenized_data = [
        tokenizer.encode(text, max_length=max_length, truncation=True, padding="max_length") for text in data
    ]

    return tokenized_data


def main():
    # 모델 이름과 데이터 파일 설정: 사용할 모델과 데이터 파일의 경로를 설정
    model_name = "gpt2"  # 여기서는 "gpt2" 모델을 사용합니다.
    data_file = '../data/processed_data.json'  # 데이터 파일 경로

    model, tokenizer = load_model_and_tokenizer(model_name)    # 모델과 토크나이저 로드
    model = prepare_model_for_instruction_tuning(model)    # Instruction Tuning을 위해 모델 학습 준비
    tokenized_data = load_and_tokenize_data(tokenizer, data_file)    # 데이터 로드 및 토큰화: 텍스트 데이터를 모델이 학습할 수 있는 형식으로 준비

    # 학습 파라미터 설정: 모델 학습의 속도와 배치 크기 설정
    learning_rate = 5e-5  # 학습 속도를 조절하는 학습률
    batch_size = 8  # 한 번에 처리할 데이터의 개수
    # TODO: torch.optim.SGD를 사용하여 옵티마이저를 설정하세요

    """
    참고:
    옵티마이저에 모델의 학습 가능한 파라미터만을 넣어 업데이트를 진행해도 되지만, 이미 requires_grad를 false로 설정하였기 때문에
    optimizer에 전체 parameter를 넣어주는 것과 optimizer에 학습할 파라미터만 넣어주는것은 같은 역할을 하게 됩니다.

    다음 예시는 학습할 파라미터만을 넣어주는 것의 예시입니다.
    옵티마이저는 모델의 학습 가능한 파라미터를 업데이트하는 역할을 합니다.
    torch.optim.SGD 클래스를 사용하여 옵티마이저를 생성합니다.

    - 첫 번째 인자: 학습 가능한 파라미터(iterable)
      - model.parameters()를 사용하면 모든 파라미터를 가져오지만, 우리는 학습 가능한(즉, requires_grad=True인) 파라미터만 선택해야 합니다.
      - 이를 위해 filter() 함수를 사용하여 조건에 맞는 파라미터만 선택합니다.
      - 예: filter(lambda p: p.requires_grad, model.parameters())

    - lr 파라미터: 학습률(learning rate)을 지정합니다.

    참고:
    - PyTorch 옵티마이저 사용법: https://pytorch.org/docs/stable/optim.html#torch.optim.SGD
    - filter() 함수 사용법: https://docs.python.org/3/library/functions.html#filter    
    """
    # 코드:
    #   optimizer = torch.optim.SGD(
    #         filter(lambda p: p.requires_grad, model.parameters()),
    #         lr=learning_rate
    #    )

    # optimizer에 전체 파라미터를 넣어주어 학습을 진행합니다.
    optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
    )

    print(f"학습 준비 완료: 학습률={learning_rate}, 배치 크기={batch_size}, 최적화 방법=SGD")

if __name__ == "__main__":
    main()
