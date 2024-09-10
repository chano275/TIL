# 농작물 수확하기 D3

T = int(input())
for test_case in range(1, T + 1):
    n = int(input())
    farm = [list(map(int, input())) for _ in range(n)]
    sum_farm = 0

    for j in range(n):
        if j == n // 2:
            #                 print(sum(farm[j]))
            sum_farm += sum(farm[j])

        elif 0 <= j < n // 2:  # 위
            #                 print(sum(farm[j][n//2 - j : n//2 + j + 1]))
            sum_farm += sum(farm[j][n // 2 - j: n // 2 + j + 1])
            # j == 0 ~ 2 > i == n//2 - j ~  n//2 + j

        else:  # 밑 j n//2 + 1  ~  n - 1
            #                 print(sum(farm[j][n//2 - (n - j - 1) : n//2 + (n - j - 1) + 1]))
            sum_farm += sum(farm[j][n // 2 - (n - j - 1): n // 2 + (n - j - 1) + 1])

    print(f'#{test_case} {sum_farm}')
