# 문제 2: 가장 긴 단어 찾기

# data/data.txt 파일을 읽기 모드("r")로 열고, file 변수에 저장


# 가장 긴 단어와 그 단어가 있는 줄 번호를 저장할 변수 초기화
longest_word = ""
longest_word_line = 0

# enumerate 함수를 사용하여 lines의 각 요소와 인덱스를 i, line으로 하나씩 꺼내서 반복문을 실행
# 만약 가장 긴 단어가 2개라면 먼저 발견된 단어가 선택됨


# 중간 결과 출력
# 가장 긴 단어와 그 단어가 있는 줄 번호 출력
print(f"가장 긴 단어: {longest_word}")
print(f"{longest_word_line}번째 줄에서 발견됨")

# 결과를 파일에 저장
# data/last_word.txt 파일에 가장 긴 단어와 그 단어가 있는 줄 번호를 저장
# open 함수를 사용하여 data/longest_word.txt 파일을 쓰기 모드("w")로 열고, file 변수에 저장


