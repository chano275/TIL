# 쇠막대기 자르기 D4


T = int(input())
for test_case in range(1, T+1):
    metal = input()

    check = []
    ans = 0

    for i in range(len(metal)): # i가 현재 보는 요소의 idx 찍고 있음

        # 예외처리
        if metal[0] == ')':
            break

        # 정상적인 0번째 원소
        else:

            # 레이저든 아니든 '(' 는 일단 stack 에 push
            if metal[i] == '(':
                check.append(metal[i])

            # ')' 들어오면 레이저인지 / 쇠막대기 끝인지 판단
            else:

                # 예외처리
                if '(' not in check:
                    break

                # 논리 생각
                else:

                    # 레이저 유무 확인, 들어있는 쇠 봉 수 == check 에 들어있는 ( 의 개수.
                    if metal[i-1] + metal[i] == '()':
                        check.pop()
                        ans += len(check)

                    # 레이저 아니라 봉 끝이었다면,
                    else:
                        check.pop()
                        ans += 1

    print(f'#{test_case} {ans}')