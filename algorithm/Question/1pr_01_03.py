# 초심자의 회문 검사 D2

def pali(str_input, idx):
    if idx <= len(str_input) // 2 - 1:  # 앞뒤 단어 하나 비교할 if문
        if str_input[idx] == str_input[-1 * idx - 1]:  # 비교해야 할 2개 글자 비교
            # print(str_input[idx], str_input[-1 * idx - 1])
            return pali(str_input, idx + 1)  # 다음 글자 비교
        else:
            return 0
    return 1


T = int(input())
for test_case in range(1, T + 1):
    s = input()
    print(f'#{test_case} {pali(s, 0)}')
