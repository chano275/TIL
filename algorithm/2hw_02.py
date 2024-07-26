# [S/W 문제해결 기본] 7일차 - 암호생성기 D3


from collections import deque
 
for _ in range(1,10+1):
    test_case = int(input())
    a = map(int, input().split())# 8개
    q = deque(a)
    chk = 0
    while 1:
        for i in range(1,6):
            q[0] -= i # 1 ~ 5
 
            if q[0] <= 0:
                q.popleft()
                q.append(0)
                chk = 1
                break
 
            else:q.append((q.popleft()))
 
        if chk == 1: break
 
    print(f"#{test_case}", end = ' ')
    for j in range(len(q)): print(q[j], end = ' ')
    print('')
    