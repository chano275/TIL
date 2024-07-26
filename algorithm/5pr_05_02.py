# [S/W 문제해결 응용] 4일차 - 보급로 D4

import sys
sys.stdin = open('5pr_05_02.txt')



from collections import deque
def bfs(road, xy):  # 0,0 ~ xy-1,xy-1
    dxy = [[1,0], [0,1], [-1,0], [0,-1]]
    queue = deque([(0,0)])

    dist = [[float("inf")] * n for _ in range(n)]  # 원점 ~ 각 점까지의 최소 거리를 저장하는 배열 
    dist[0][0] = 0                                 # 맨 처음값 0,0 은 거리 값이 0이기에 dist 배열에 0 저장

    while queue:
        x, y = queue.popleft()

        for dx, dy in dxy:
            nx, ny = x + dx, y + dy

            if nx < 0 or nx >= n or ny < 0 or ny >= n:continue   # 범위 나가면 컷


            # 기존에 고려했던 탈출조건.
            # 맨 마지막 좌표에 도착했을 때에 해당 좌표의 위 아래까지 오는 최소값을 비교했는데
            # 도착만 하면 탈출하는 조건문이라 사실상 필요가 없었음

            # if nx == xy-1 and ny == xy-1: 
            #     return min(dist[xy-2][xy-1], dist[xy-1][xy-2])



            # 본 코드는 visited 를 사용하지 않아서, 정말 고려할 수 있는 모든 경로로 움직이는데,
            # 모든 경로를 지날때 최소 거리를 dist 배열에 저장한다고 했으므로
            # 해당 거리의 최소 값에서 다음 칸의 값을 더해주어 min값을 구해
            # 모든 점의 최소 거리를 구함 
            # 값이 변한다면 해당 점을 지나는 경우를 다시 생각해줘야 하므로 append (이 알고리즘 통해 시간줄이기)
            t1 = dist[nx][ny]
            dist[nx][ny] = min(dist[nx][ny], road[nx][ny] + dist[x][y])
            t2 = dist[nx][ny]

            if t1 != t2: queue.append((nx, ny))

    # 정말 모든 경우를 체크하고 저장되어 있는 최소값(도달을 위한) 저장 
    return dist[xy-1][xy-1]

T = int(input())
for test_case in range(1, T+1):
    n = int(input())
    road = [list(map(int, input())) for _ in range(n)]

    print(f'#{test_case} {bfs(road, n)}')

    # break