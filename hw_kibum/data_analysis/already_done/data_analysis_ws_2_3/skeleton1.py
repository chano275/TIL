# 문제 1: Vector2D 클래스 정의 및 연산자 오버로딩

class Vector2D:
    def __init__(self, x, y):    # 생성자
        self.x = x
        self.y = y

    # 연산자 오버로딩
    def __add__(self, other):    # 두 벡터의 합을 반환
        return Vector2D(self.x + other.x, self.y + other.y)
    def __sub__(self, other):    # 두 벡터의 차를 반환
        return Vector2D(self.x - other.x, self.y - other.y)
    def __mul__(self, scalar):    # 벡터와 스칼라의 곱을 반환
        return Vector2D(self.x * scalar, self.y * scalar)

    # 객체를 문자열로 표현
    def __str__(self):
        return f'({self.x}, {self.y})'


# 객체 생성 및 연산자 오버로딩 사용
sensor1_vector = Vector2D(1, 2)
sensor2_vector = Vector2D(3, 4)

# 두 벡터의 합, 차, 스칼라 곱 계산
combined_vector = sensor1_vector.__add__(sensor2_vector)
difference_vector = sensor1_vector.__sub__(sensor2_vector)
scaled_vector = sensor1_vector.__mul__(2)

# 결과 출력
print("문제 1. Vector2D 클래스 정의 및 연산자 오버로딩")
print("두 벡터의 합, 차, 스칼라 곱 계산 결과")
print("두 벡터의 합:", combined_vector)
print("두 벡터의 차:", difference_vector)
print("벡터의 스칼라 곱:", scaled_vector)




