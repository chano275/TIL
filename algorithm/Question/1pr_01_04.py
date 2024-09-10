# 단순 2진 암호코드 D3

def checker(li):
    for i in range(10):
        if li == numbers[i]:
            return i


T = int(input())
for test_case in range(1, T + 1):
    n, m = map(int, input().split())  # 배열 세로 / 가로
    scanner = [list(map(int, input())) for _ in range(n)]

    numbers = [[0, 0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 1], [0, 1, 1, 1, 1, 0, 1],
               [0, 1, 0, 0, 0, 1, 1], [0, 1, 1, 0, 0, 0, 1], [0, 1, 0, 1, 1, 1, 1], [0, 1, 1, 1, 0, 1, 1],
               [0, 1, 1, 0, 1, 1, 1], [0, 0, 0, 1, 0, 1, 1]]

    start_j = -1
    where_is_one = -1
    for j in range(n):
        for i in range(m - 1, -1, -1):  # 뒤에서부터 확인
            if scanner[j][i] == 1:
                start_j = j
                where_is_one = i
                break

    code = scanner[start_j][where_is_one - 55: where_is_one + 1]
    decoder = []

    for q in range(8):  # q : 0 ~ 7
        decoder.append(checker(code[7 * q: 7 * (q + 1)]))  # 0 ~ 5 / 7 ~ 15 / ...

    sum_z, sum_h = 0, 0

    for a in range(len(decoder)):
        if a % 2 == 1:
            sum_z += decoder[a]
        else:
            sum_h += decoder[a]

    if (sum_h * 3 + sum_z) % 10 == 0:
        print(f'#{test_case} {sum(decoder)}')
    else:
        print(f'#{test_case} 0')
