# 문제 3: 벡터 내적 계산 메서드 추가
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
    def __mul__(self, scalar):    # 벡터와 스칼라의 곱을 반환
        return Vector2D(self.x * scalar, self.y * scalar)
    def length(self):  
        return math.sqrt(self.x**2 + self.y**2)
    def __str__(self):    # 객체를 문자열로 표현하는 메서드 정의
        return f"({self.x}, {self.y})"

    def dot_product(self, other):    # 벡터의 내적을 반환하는 메서드 추가로 정의
        return self.x * other.x + self.y * other.y


# 객체 생성 및 내적 계산 예제
sensor1_vector = Vector2D(1, 2)
sensor2_vector = Vector2D(3, 4)

# 두 벡터의 내적 계산
answer = sensor1_vector.dot_product(sensor2_vector)

# 결과 출력
print("문제 3. 벡터 내적 계산 메서드 추가")
print("두 벡터(sensor1_vector, sensor2_vector)의 내적:", answer)
