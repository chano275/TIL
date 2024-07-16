# 파이썬으로 git 명령어를 사용할 예정.
# 모듈 2가지 os, subprocess

# os
# 현재 작업 위치
import os
import subprocess
# 현재 폴더 경로를 변수에 저장
current_folder = os.getcwd()
# 특정 폴더 내로 이동 후에 작동하도록 하려면? os 모듈의 어떤 함수?

# 현재 폴더 및 하위 폴더를 반복
# subprocess.run(['문자열 형태로', '실행할 명령어', 변수])
subject = input('과목을 입력해 주세요 : ')
seperators = ['hw', 'ws']  # -> 반복문 사용해서 만들어 지도록 하면되겠다.
set_number = input('세트 번호를 입력해 주세요 : ')
# 보충 수업에서 쓰는 문제는 a, b, c로 되어 있다.
stages = range(1, 6) # hw, ws 에 따라 달라져야겠다.
for sep in seperators:
    for stage in stages:
        URL = f'https://lab.ssafy.com/chano01794/{subject}_{sep}_{set_number}_{stage}'
        print(URL)
        subprocess.run(['git', 'clone', URL])
'''
https://lab.ssafy.com/계정ID/{과목}_{구분}_{세트번호}_{단계}
과목 : python, web, js, django, db, vue
구분 : hw, ws
세트번호 : 과목마다 다르다 -> 클론 받고자 하는 대상의 set번호 확인
단계 : 1~5 (과제: 2, 4)

실습실에 생성되는 repository마다 URL이 서로 다른데... 어떻게 사용하느냐?
string intertpolation
$ git clone URL
'''
