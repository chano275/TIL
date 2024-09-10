# 거듭 제곱 D3

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