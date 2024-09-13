# 문제 2: 벡터 길이 계산 메서드 추가
import math  # math 모듈을 import하기


class Vector2D:
    # 생성자 정의
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # 연산자 오버로딩
    def __add__(self, other):    # 두 벡터의 합을 반환하는 메서드 정의
        return Vector2D(self.x + other.x, self.y + other.y)
    def __sub__(self, other):    # 두 벡터의 차를 반환하는 메서드 정의
        return Vector2D(self.x - other.x, self.y - other.y)
    def __mul__(self, scalar):    # 벡터와 스칼라의 곱을 반환하는 메서드 정의
        return Vector2D(self.x * scalar, self.y * scalar)

    # 백터의 길이를 반환하는 메서드 정의
    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    # 객체의 문자열을 반환하는 메서드 정의
    def __str__(self):
        return f"({self.x}, {self.y})"

# 객체 생성 및 길이 계산
sensor1_vector = Vector2D(2,3)
print("문제 2. 벡터 길이 계산 메서드 추가")
print("객체 sensor1_vector의 길이:", sensor1_vector.length())
