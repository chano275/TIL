# [S/W 문제해결 기본] 7일차 - 미로1 D4


#import sys
#sys.stdin = open('5hw_05_01.txt')
from collections import deque


def bfs(miro, ):
    dxy = [[1,0], [0,1], [-1,0], [0,-1]]
    queue = deque([(1,1)])
    visited = [[-1] * 16 for _ in range(16)]
    visited[1][1] = 0

    while queue:
        x, y = queue.popleft()
        for dx, dy in dxy:
            nx, ny = dx + x, dy + y
            
            if nx < 0 or nx >= 16 or ny < 0 or ny >= 16:continue

            if visited[nx][ny] != -1: continue

            if miro[nx][ny] == 1: continue

            if miro[nx][ny] == 3: return 1

            queue.append((nx, ny))
            visited[nx][ny] = 0

    return 0






T = 10
for test_case in range(1, T+1):
    tc = int(input())
    miro = [list(map(int, input())) for _ in range(16)]
    print(miro)
    print(f'#{test_case} {bfs(miro)}')