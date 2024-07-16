# 아래에 코드를 작성하시오.
numbers = [1,2,3,4,5,6,7,8,9,10]
for i in range(len(numbers)):
    if numbers[i] == 5:
        break
    if numbers[i] % 2 == 0:
        print(numbers[i])
    else:
        print(f'{numbers[i]}은(는) 홀수')
    