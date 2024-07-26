"""
4×4 크기의 격자판 > 각 칸에 0 ~ 9

임의의 위치에서 시작 > 동서남북 네 방향 인접한 격자로 6번 이동 > 7자리의 수

O : 한 번 거쳤던 격자칸 다시 // 0으로 시작
X : 격자판을 벗어나는 이동

만들 수 있는 서로 다른 일곱 자리 수들의 개수


[입력]
첫 번째 줄에 테스트 케이스의 수 T가 주어진다.
각 테스트 케이스마다 4개의 줄에 걸쳐서, 각 줄마다 4개의 정수로 격자판의 정보가 주어진다.

[출력]
각 테스트 케이스마다 ‘#x ’(x는 테스트케이스 번호를 의미하며 1부터 시작한다)를 출력하고,
격자판을 이동하며 만들 수 있는 서로 다른 일곱 자리 수들의 개수를 출력한다.

"""

# 재귀 : 문자열이 7개 되었을 때에 탈출
# x, y를 옮기는 법..?
# 갔다가 돌아오는것도 생각..?
import time
import sys

sys.stdin = open('4pr_04_01.txt')

start_time = time.time()

dx = [1, 0, -1, 0]
dy = [0, 1, 0, -1]

def dfs(checker, x, y, memo, str_):
    if len(str_) == 7:
        if str_ not in memo:
            memo.append(str_)
        return

    if 0 <= x < 4 and 0 <= y < 4:
        str_ += str(checker[x][y])
        for chk in range(4):
            if 0 <= x + dx[chk] < 4 and 0 <= y + dy[chk] < 4:
                dfs(checker, x + dx[chk], y + dy[chk], memo, str_)


T = int(input())
for test_case in range(1, T + 1):
    checker = []
    for _ in range(4):
        checker.append(list(map(int, input().split())))

    memo = []  # str_ 7 되었을 때에 넣을 memo list
    str_ = ''

    for i in range(4):
        for j in range(4):
            dfs(checker.copy(), i, j, memo, str_)

    print(f'#{test_case} {len(memo)}')


end_time = time.time()

#print(end_time - start_time)