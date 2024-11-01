# 사용자 정의 예외 클래스 생성

# 문제 1. InvalidValueError 라는 사용자 정의 예외 클래스를 정의 / ValueError 클래스를 상속받아서 InvalidValueError 클래스를 정의
class InvalidValueError(ValueError):
    def __init__(self, value):
        super().__init__()
        self.value = value

def check_positive_value(value):# 문제 2. 0보다 작은 값이 들어올 경우 InvalidValueError 예외를 발생
    if value < 0:
        print(f'{value}는 이상')
        raise InvalidValueError(value)
    else:
        print(f'{value}는 유효')

try:
    check_positive_value(1)
    check_positive_value(-1)
except InvalidValueError as e:
    print(e)
