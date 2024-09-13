# 문제 1: 단어 수 계산 후 파일에 저장

# 텍스트 파일의 내용을 읽어들이기 / open 함수를 사용하여 data/data.txt 파일을 읽기 모드("r")로 열고, file 변수에 저장
with open("C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_ws_2_5/data.txt", "r") as file:

# 각 줄의 단어 수 계산 / 반복문을 사용하여 lines의 각 요소를 line으로 하나씩 꺼냄.
# split 함수를 사용하여 단어로 분리한 후 len 함수를 사용하여 단어 수를 계산하여 word_counts 리스트에 저장
    lines = file.readlines()
    word_counts = [len(line.split()) for line in lines]

# 중간 결과 출력 / enumerate > word_counts의 각 요소와 인덱스를 i, count로 하나씩 꺼내서 출력
# f-string을 사용하여 i+1번째 줄: count개의 단어 형식으로 출력
for i, count in enumerate(word_counts):
    print(f'{i+1}번째 줄에 {count}개 단어')

# 결과를 파일에 저장
# open 함수를 사용하여 data/word_count.txt 파일을 쓰기 모드("w")로 열고, file 변수에 저장
# 반복문을 사용하여 word_counts의 각 요소와 인덱스를 i, count로 하나씩 꺼내서 파일에 쓰기
with open("C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_ws_2_5/word_count.txt", "w") as file:
    for i, count in enumerate(word_counts):
        file.write(f'{i+1}번째 줄에 {count}개 단어\n')


