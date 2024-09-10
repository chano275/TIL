# 계산기1 D4


# stack 으로 다시 풀기 ? 

for test_case in range(1, 11):
    s_len = int(input())
    s = list(map(int, input().split('+')))
    print(f'#{test_case} {sum(s)}')