# 중위순회 D4

import sys
sys.stdin = open('hw_03.txt')

def in_order(node):
    if node: 
        in_order(data[node][2])
        print(data[node][1], end = '')
        in_order(data[node][3])

for tc in range(1,5):
    N = int(input())   
    # 맵 자체가 각각의 객체에 func 적용시킴
    data = [list(map( lambda x: int(x) if x.isdecimal() else x, input().split())) for _ in range(N)] # input.split 실행되며 리스트로 받아지고, map 함수 자체가 func를 각각의 인자들에 적용시킨다. func 가 lambda x 

    for arr in data:
        while len(arr) != 4:
            arr.append(0)

    data.insert(0, [0,0,0,0])
    print(f'#{tc} ', end = '')
    in_order(1)
    print('')