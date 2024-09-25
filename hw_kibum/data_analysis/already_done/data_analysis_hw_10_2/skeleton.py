import pandas as pd

# 카이사르 암호 함수
def caesar_cipher(text, shift):
    result = []
    for char in text:
        if char.isalpha(): # 영어 알파벳인지 확인
            base = ord('A') if char.isupper() else ord('a')
            shifted = chr((ord(char) - base + shift) % 26 + base)
            result.append(shifted)
        else:
            result.append(char)  # 알파벳이 아니면 그대로 추가
    return ''.join(result)

# 예제 데이터 (stationName을 영어로 구성)
df = pd.DataFrame({
    'stationName': ['Seoul Station', 'Gangnam Station', 'Jongno3ga']
})

# 정류소 이름 암호화 (영어만 처리)
df['암호화된_정류소_이름'] = df['stationName'].apply(lambda x: caesar_cipher(x, 3)) # 암호와에 사용할 키: 3

# 결과 출력
print(df[['stationName', '암호화된_정류소_이름']])
