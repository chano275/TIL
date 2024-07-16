# 아래에 코드를 작성하시오.

class Theater():
    def __init__(self, name, total_seats, reserved_seats):
        self.name = name
        self.total_seats = total_seats
        self.reserved_seats = reserved_seats
    
    def reserve_seat(self):
        if self.reserved_seats < self.total_seats:
            self.reserved_seats += 1
            print('좌석 예약이 완료되었습니다.')
        else:
            print('좌석 예약에 실패했습니다.')

class MovieTheater(Theater): # 상속받은 class의 init 변함 없으면, 선언 안해도 괜찮다
    total_movies = 0

    @classmethod
    def add_movie(cls):
        cls.total_movies += 1
        print('영화가 성공적으로 추가되었습니다.')
    
    @staticmethod
    def description(theater_):
        print(f'영화관 이름: {theater_.name}')
        print(f'총 좌석 수: {theater_.total_seats}')
        print(f'예약된 좌석 수: {theater_.reserved_seats}')
        print(f'총 영화 수: {MovieTheater.total_movies}')

        # 정적 메서드에 인스턴스를 주어 출력하는 식의 동작 가능 / SELF X 

a = Theater('메가박스', 100, 5)
b = MovieTheater('CGV', 200, 10)

a.reserve_seat()
a.reserve_seat()
b.add_movie()
b.description(b)