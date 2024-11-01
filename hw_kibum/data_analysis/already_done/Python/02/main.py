# 해당 문제를 풀기 위한 포함 기술스택들을 작성해주세요: python, calculator
# 이 스크립트는 calculator 패키지의 모듈을 사용하여 기본 연산(덧셈, 뺄셈, 곱셈, 나눗셈)을 수행합니다.
# calculator 패키지에서 각 모듈을 임포트합니다.
# 각 연산을 수행하고 결과를 저장합니다.

from calculator import add, divide, multiply, subtract


print("과제 2. 계산기 프로그램")
print(f"10 + 5 = {add.add(10, 5)}")
print(f"10 - 5 = {subtract.subtract(10, 5)}")
print(f"10 * 5 = {multiply.multiply(10, 5)}")
print(f"10 / 5 = {divide.divide(10, 5)}")
