# 아래에 코드를 작성하시오.
print('현재 좌석')

a = [['O','O','X'], ['X','O','X'], ['X','O','X']]

for j in range(3):
    for i in range(3):
        print(a[j][i], end = ' ')
    print('')