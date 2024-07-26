# [모의 SW 역량테스트] 탈주범 검거



import sys
sys.stdin = open('5pr_05_01.txt')
###############################################################
from collections import deque

def nxy(t_type):
    if t_type == 1:        dxy = [[1,0], [0,1], [-1,0], [0,-1]]
    elif t_type == 2:        dxy = [[1,0], [-1,0]]
    elif t_type == 3:        dxy = [[0,1], [0,-1]]
    elif t_type == 4:        dxy = [[-1,0], [0,1]]
    elif t_type == 5:        dxy = [[1,0], [0,1]]
    elif t_type == 6:        dxy = [[1,0], [0,-1]]
    elif t_type == 7:        dxy = [[-1,0], [0,-1]]

    return dxy


def bfs(tunnel, x,y, time):
    queue = deque([(x,y)])
    visited = [[float('inf')] * m for _ in range(n)]
    visited[x][y] = 1

    while queue:
        cx, cy = queue.popleft()
        dxy_ = nxy(tunnel[cx][cy])
        
        for dx, dy in dxy_:
            nx , ny = cx + dx, cy + dy
            
            if nx < 0 or nx >= n or ny < 0 or ny >= m: continue
            if tunnel[nx][ny] == 0:  continue 
            
            visited[nx][ny] = min(visited[cx][cy] + 1,visited[nx][ny])
            if nx == r and ny == c:  
                if visited[nx][ny] <= time:
                    return 1
                return -1
            queue.append((nx, ny))
    return -1


T = int(input())
for test_case in range(1, T+1):
    n, m, r, c, l = list(map(int, input().split()))
    # 터널 세로 / 가로 // 맨홀 세로 / 가로 / 시간 
    tunnel = [list(map(int, input().split())) for _ in range(n)]

    ans = 0

    print(f'{l} 시간 안에 도달해야 한다')

    for i in range(n):  
        for j in range(m):
            if tunnel[i][j] != 0:
                if i==r and j == c: ans += 1
                else:
                    if bfs(tunnel, i,j, l) == True:
                        # print(i, j)
                        # print('*********')
                        ans += 1 

    print(f'#{test_case} {ans}')

    # break