from transformers import PreTrainedTokenizerFast

# 1. 토크나이저 로드
def load_tokenizer(model_name='skt/kogpt2-base-v2'):
    """
    사전 학습된 GPT-2 토크나이저를 로드하고, 필요시 EOS 토큰과 패딩 토큰을 설정하는 함수
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    
    # 참고: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.add_special_tokens
    # EOS 토큰이 설정되지 않은 경우, 새로운 EOS 토큰을 추가
    # TODO: <eos> 형태로 'eos_token'을 추가해주세요.
    if tokenizer.eos_token is None:
        tokenizer._____________________
    
    # 패딩 토큰이 설정되지 않은 경우, EOS 토큰을 패딩 토큰으로 설정
    # TODO: 'pad_token'으로 eos_token을 매핑해주세요.
    if tokenizer.pad_token is None:
        tokenizer._____________________
    
    return tokenizer

def main():
    tokenizer = load_tokenizer(model_name='skt/kogpt2-base-v2')
    print(f"tokenizer: {tokenizer}")
    print("토크나이저가 성공적으로 로드되고 설정되었습니다.")

if __name__ == "__main__":
    main()
