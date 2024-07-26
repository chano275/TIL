# 이진수 표현 D3


T = int(input())
for test_case in range(1, T+1):
    n, m = map(int, input().split())

    # m의 마지막 n개의 비트가 모두 1이라면 On
    if (2**n - 1) & m == 2**n - 1:
        print(f'#{test_case} ON')
    else:
        print(f'#{test_case} OFF')
