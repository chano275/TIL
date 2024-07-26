# [S/W 문제해결 기본] 4일차 - 거듭 제곱 D3


## 0승이 1인걸 생각해서 한번 더 짜보기 

def gd(step, a, b):
    if b == 1:
        return a
    else:
        return gd(step, a * step, b - 1)
 
while 1:
    try:
        test_case = int(input())
        n, m = map(int, input().split())
        print(f"#{test_case} {gd(n, n, m)}")
    except:
        break