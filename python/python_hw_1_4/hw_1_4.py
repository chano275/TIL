# '=' 포커싱 후 Ctrl + D > 동일한 단어 찾고 변환도 가능
# 학생 점수 정보
A = {   "Alice" : 85,
   "Bob" : 78,
   "Charlie" : 92,
   "David" : 88,
   "Eve" : 95}

# 아래에 코드를 작성하시오.

student = {"Alice" : 85,   "Bob" : 78,   "Charlie" : 92,   "David" : 88,   "Eve" : 95}

#################################################

print('1. 학생들의 이름과 점수를 딕셔너리에 저장')
print(f'students type: {type(student)}')
print(f'학생들의 이름과 점수: {student}')

#################################################

py = sum(student.values()) / len(student)
print(f'2. 모든 학생의 평균 점수: {py:.2f}') ## 출력시 형태 

#################################################
# 튜플로 만든 list < value가 80 이상이면 
# list comprehension ! 
# top_students = [(key, value) for key, value in student.items() if value >= 80]
# print(top_students)

ans = []
print('3. 기준 점수(80점) 이상을 받은 학생 수:', end = ' ')
for k,v in student.items():
    if v >= 80:
        ans.append(k)
print(ans)

#################################################

print('4. 점수 순으로 정렬: ')
sorted_d = sorted(student.items(), key=lambda x: x[1], reverse=True)
# sorted : 원래 key 값 기준으로 정렬됨. 
#          첫번째 인자 : 반복 가능한 객체 / dict가 가지고 있는 키와 value 다 필요하므로 items 가져옴 
# key : 무엇을 기준으로 정렬 ? > value  >> 람다 사용법 ㄹㄹㄹ 


for q in sorted_d:
    print(f'{q[0]}: {q[1]}')

#################################################

diff = max(student.values()) - min(student.values())
print(f'5. 점수가 가장 높은 학생과 가장 낮은 학생의 점수 차이: {diff}')

#################################################

print('6. 각 학생의 점수가 평균보다 높은지 낮은지 판단: ')
for k,v in student.items():
    if v <= py:
        print(f'{k} 학생의 점수 ({v})는 평균 이하입니다.')
    else:
        print(f'{k} 학생의 점수 ({v})는 평균 이상입니다.')