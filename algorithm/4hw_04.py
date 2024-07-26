# [모의 SW 역량테스트] 숫자 만들기

## 풀이 : 
"""
문제를 보고 우선적으로 떠올려야 하는 것 : 순열 
n = 12 < 12! 꽤 커서 DFS로 순열 구하는 생각 해야 함 

1. 숫자 고르고
2. 연산자 고르고
3. 다음 숫자를 골라라 

dfs(1. 재귀호출 중단시킬 요소 숫자의 idx , 2. 누적해서 결과 가져갈 요소 res , 3. 연산자 개수 배열 ... )

oper에서 하나 골라오면 숫자 -1 시킴 
"""
import sys
sys.stdin = open('4hw_04.txt')
T = int(input())
for test_case in range(1, T+1):
    n = int(input())
    op_input_list = list(map(int, input().split())) # + - * //   개수 든 숫자 배열
    number_list = list(map(int, input().split()))

    # 문제에서 원하는 수적 결과의 최소, 최대값을 구해야 한다.
    max_num = -100000000
    min_num = 100000000

    # idx : 이번에 우리가 선택한 숫자
    # res : 여태까지 ㄴ투적된 결과
    # op_list : 남은 연산자 ( 사실상 visited / TF가 아닌 숫자로 취급되고 0이 되면 VISITED = FALSE 와 동일하다 )
    def create_num(op_list, idx, res):
        global max_num, min_num
        if idx == n:  # 모든 숫자 다 계산했다면 결과 반영 
            max_num = max(max_num, res)
            min_num = min(min_num, res)
            return 
        
        # idx는 i / cnt 는 남은 연산자 개수 
        for op_idx, op_cnt in enumerate(op_list):
            if op_cnt == 0:continue

            tmp_res = res

            if op_idx == 0:
                tmp_res += number_list[idx]
            elif op_idx == 1:
                tmp_res -= number_list[idx]
            elif op_idx == 2:
                tmp_res *= number_list[idx]
            elif op_idx == 3:
                tmp_res = int(tmp_res / number_list[idx])

            op_list[op_idx] -= 1
            create_num(op_list, idx+1, tmp_res)
            op_list[op_idx] += 1 # 이게 메인... 숙서를 섞나 

        


    # DFS 구현에 있어서 중요한 것
    # 1. 재귀호충ㄹ을 중단할 파라미터 ( 어떤 숫자를 선택하고 있는지 / IDX )
    # 2. 누적해서 가져갈 결과 파라미터 ( 여태까지 수식에서 나온 결과를 전달 ) 
    # 연산자 목록 = VISITED 

    init_num = number_list[0]
    init_idx = 1
    # 문제에서 주어진 숫자의 순서 변경 불가. 
    create_num(op_input_list, init_idx, init_num)
    print(f'#{test_case} {max_num - min_num}')