# [S/W 문제해결 기본] 9일차 - 사칙연산 D4


# node 통해서 트리로 풀기 

for test_case in range(1, 11):
    n = int(input())

    operator = ['+', '-', '/', '*']
    tree = [[0] * 3 for _ in range(n+1)]

    for inp in range(n):
        v_info = (list(map(str, input().split())))
        if v_info[1] in operator:   # 왼쪽자식 / 오른쪽자식 // idx 1부터 봐
            tree[inp+1][0] = (v_info[1])
            tree[inp+1][1] = int(v_info[2])
            tree[inp+1][2] = int(v_info[3])

        else:   # 노드에 숫자만 넣어주면 된다
            tree[inp+1][0] = int(v_info[1])

    for test in range(n, 0, -1):
        if tree[test][0] in operator:
            if tree[test][0] == operator[0]:    # +
                tree[test][0] = tree[(tree[test][1])][0] + tree[tree[test][2]][0]
            elif tree[test][0] == operator[1]:  # -
                tree[test][0] = tree[(tree[test][1])][0] - tree[tree[test][2]][0]
            elif tree[test][0] == operator[2]:  # /
                tree[test][0] = tree[(tree[test][1])][0] / tree[tree[test][2]][0]
            else:
                tree[test][0] = tree[(tree[test][1])][0] * tree[tree[test][2]][0]

    print(f'#{test_case} {int(tree[1][0])}')

