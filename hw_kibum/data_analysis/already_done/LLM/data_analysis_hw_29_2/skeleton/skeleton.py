import pandas as pd
import json


def preprocess_data(file_path):  # 데이터 전처리 함수
    df = pd.read_csv(file_path)      # 데이터 불러오기
    selected_columns = ['날짜', '종가', '거래량']      # 필요한 컬럼만 선택
    df = df[selected_columns]

    # TODO : dropna() 메서드를 사용하여 결측치를 제거하세요.
    # dropna() 메서드 참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
    df = df.dropna()      # 결측치 제거 - dropna() 메서드를 사용하여 결측치를 제거합니다.

    # TODO : drop_duplicates() 메서드를 사용하여 중복된 데이터를 제거하세요.
    # drop_duplicates() 메서드 참고: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html
    df = df.drop_duplicates()      # 중복된 데이터 제거 - drop_duplicates() 메서드를 사용하여 중복된 리이터를 제거합니다.

    return df


def chunk_text(data, chunk_size=256):  # 데이터를 청크 단위로 분할하는 함수
    """
    텍스트 데이터를 청크 단위로 분할합니다.
    chunk_text() vs RecursiveCharacterTextSplitter:

    chunk_text() 장점:
    - 단순하고 빠르게 구현 가능, 초심자에게 적합
    - 추가 라이브러리 필요 없음
    chunk_text() 단점:
    - 중첩 불가, 문맥 유지 어려움

    RecursiveCharacterTextSplitter 장점:
    - 청크 중첩 허용, 문맥 유지 가능
    - 고급 텍스트 분할에 유리
    RecursiveCharacterTextSplitter 단점:
    - 설정 복잡, 추가 라이브러리 필요

    결론:
    - 과제1에서는 기본적인 청크화 개념을 익히기 위해 chunk_text()를 사용합니다.
    RecursiveCharacterTextSplitter() 참고: https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
    """

    # 청크 단위로 데이터를 분할하기 위해 청크 리스트와 현재 청크를 초기화합니다.
    chunks = [] # 청크 리스트
    current_chunk = [] # 현재 청크

    for index, row in data.iterrows():    # 데이터프레임을 반복하면서 텍스트 데이터를 청크 단위로 분할하고 청크 리스트에 추가
        # f-string을 사용하여 각 행의 데이터를 포맷팅합니다.
        # row['컬럼명']을 통해 '날짜', '종가', '거래량' 값을 접근하고 문자열로 결합합니다.
        # f-string 예시: f"{변수명} - {다른 변수명}"
        # 여기서 f-string은 row['날짜'], row['종가'], row['거래량'] 값을 문자열에 삽입합니다.
        # 예를 들어, row['날짜']가 '2023-10-01', row['종가']가 1000, row['거래량']이 500일 경우,
        # text는 "2023-10-01 - 종가: 1000, 거래량: 500"이 됩니다.
        # 참고: f-string 참고: https://docs.python.org/ko/3/tutorial/inputoutput.html#formatted-string-literals
        text = f"{row['날짜']} - 종가: {row['종가']}, 거래량: {row['거래량']}"
        current_chunk.append(text)

        if len(current_chunk) >= chunk_size:        # 현재 청크의 길이가 설정된 크기를 초과할 경우, 새로운 청크로 시작
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # 마지막 청크 추가
    # 마지막 청크가 설정된 크기보다 작을 수 있으므로, 마지막 청크를 추가해줍니다.
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def save_to_json(data_chunks, output_file):  # JSON 파일로 저장하는 함수
    with open(output_file, 'w', encoding='utf-8') as json_file:    # 파일을 쓰기 모드로 열고 JSON 파일로 데이터 저장
        # TODO: json.dump()를 사용하여 데이터를 JSON 파일로 저장하세요.
        # json.dump() 함수의 매개변수 설명:
        # - 첫 번째 매개변수: 저장할 데이터 (여기서는 {"chunks": data_chunks})
        # - 두 번째 매개변수: 파일 객체 (여기서는 json_file)
        # - ensure_ascii=False: ASCII가 아닌 문자도 그대로 저장 (한글 등 비ASCII 문자를 그대로 저장)
        # - indent=4: JSON 파일을 보기 좋게 들여쓰기 (4칸 들여쓰기)
        # json.dump() 참고: https://docs.python.org/ko/3/library/json.html#json.dump
        json.dump({"chunks":data_chunks}, json_file, ensure_ascii=False, indent=4)


def main():
    input_file, output_file = '../data/financial_data.csv', 'processed_data.json'  # 파일 경로와 출력 파일 설정
    processed_data = preprocess_data(input_file)                                   # 데이터 전처리
    data_chunks = chunk_text(processed_data, chunk_size=256)                       # 텍스트 데이터를 청크로 분할
    save_to_json(data_chunks, output_file)                                         # 분할된 데이터를 JSON 파일로 저장
    print(f"데이터가 {output_file} 파일로 저장되었습니다.")


if __name__ == "__main__":
    main()
