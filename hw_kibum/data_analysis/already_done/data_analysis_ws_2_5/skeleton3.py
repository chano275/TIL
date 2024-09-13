# 문제 3: 파일 내용을 거꾸로 저장

with open("C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_ws_2_5/data.txt", "r") as file:
    lines = file.readlines()

# 거꾸로 변환된 내용을 저장할 리스트 초기화
# 슬라이싱은 문자열의 일부분을 추출하는 방법으로, [시작 인덱스:끝 인덱스:간격] 형식으로 사용합니다.
# 시작 인덱스와 끝 인덱스를 생략하면 문자열의 처음부터 끝까지 추출하며, 간격을 음수로 지정하면 문자열을 거꾸로 만들 수 있습니다.
reversed_lines = [line[::-1] for line in lines[::-1]]

# 중간 결과 출력
print("거꾸로 변환된 내용:")
print(reversed_lines)



for r in reversed_lines:# reversed_lines의 각 요소를 line으로 하나씩 꺼내서 출력
    print(r, end = '')    # end="": print 함수가 줄바꿈을 하지 않도록 설정    # 이미 각 요소의 끝에 줄바꿈이 포함되어 있기 때문에 줄바꿈을 두 번 하지 않도록 설정



# 결과를 파일에 저장

