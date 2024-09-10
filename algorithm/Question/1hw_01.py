# Ladder1 D4

def start(li, new_j, new_i):
    while new_j != 0 :
        if 0 <= new_i + 1 < 100 and li[new_j][new_i + 1] == 1:
            while new_i + 1 < 100 and li[new_j][new_i + 1] == 1:
                new_i += 1
            new_j -= 1

        elif 0 <= new_i - 1 < 100 and li[new_j][new_i - 1] == 1:
            while 0 <= new_i - 1 and li[new_j][new_i - 1] == 1:
                new_i -= 1
            new_j -= 1

        else:
            new_j -= 1

    return new_i


for test_case in range(1, 10 + 1):
    num = int(input())
    ladder = [list(map(int, input().split())) for _ in range(100)]

    for i in range(100):
        if ladder[99][i] == 2:
            print(f'#{test_case} {start(ladder, 99, i)}')
            break
