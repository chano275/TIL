# 최대 상금 D3

def dfs(change_, arr):
    global ans
    if change_ == change:  # 변환 끝
        ans = max(ans, int(''.join(arr)))
        return

    for i in range(len(arr)):
        for j in range(i+1 , len(arr)): # i+1 인 이유 ?
            arr[i], arr[j] = arr[j], arr[i]
            dfs(change_ + 1, arr)





result = []
T = int(input())
for test_case in range(1, T + 1):
    N, change = map(int, input().split())
    ans = 0
    arr = list(str(N)) # OK 문자로 받았어 일단



    dfs(0, arr)

    print(f'#{test_case} {ans}')



