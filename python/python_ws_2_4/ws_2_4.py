# 아래에 코드를 작성하시오.
class MovieTheater():
    total_movies = 0

    def __init__(self, name, total_seats):    # 생성자 내에 들어가는 인스턴스 변수 : 각 인스턴스별로 따로 관리 
        self.name = name
        self.total_seats = total_seats
        self.reserved_seats = 0

    def __str__(self):          # __str__ : 객체 자체를 출력할 때 넘겨주는 형식을 지정해주는 메서드
        return self.name        #           return을 print 와 같이 이용 가능 / f'' 도 이용 가능 

    def reserve_seat(self):
        if self.reserved_seats >= self.total_seats:
            print('좌석 예약에 실패했습니다.')   
        else:
            self.reserved_seats += 1
            print('좌석 예약이 완료되었습니다.')

    def current_status(self):
        print(f'{self.name} 영화관의 총 좌석 수: {self.total_seats}')
        print(f'{self.name} 영화관의 예약된 좌석 수: {self.reserved_seats}')

############################################################
    @classmethod
    def add_movie(cls): # total_movies 1 증가 / 영화 추가 성공 메세지 반환 
        MovieTheater.total_movies += 1
        print('영화가 성공적으로 추가되었습니다.')
        
    @staticmethod
    def description(): # 영화관 정보 출력 / 영화관의 이름, 총좌석수, 예약좌석수, 총영화수 
        print('이 클래스는 영화관의 정보를 관리하고 좌석 예약 및 영화 추가 기능을 제공합니다.')
        print("영화관의 이름, 총 좌석 수, 예약된 좌석 수, 총 영화 수를 관리합니다.")
############################################################
class VIPMovieTheater(MovieTheater):
    def __init__(self, name, total_seats, vip_seats):
        super().__init__(name, total_seats)
        self.vip_seats = vip_seats

    def reserve_vip_seat(self):
        if self.vip_seats > 0:
            self.vip_seats -= 1
            print('VIP 좌석 예약이 완료되었습니다.')
        
        else: 
            print('예약 가능한 VIP 좌석이 없습니다.')   

    def reserve_seat(self):
        if self.vip_seats > 0:
            self.reserve_vip_seat()

        else:
            super().reserve_seat()


a = MovieTheater('메가박스', 150)
b = MovieTheater('CGV', 200)
c = VIPMovieTheater('롯데시네마',300,3)

a.reserve_seat()
b.reserve_seat()
b.reserve_seat()
c.add_movie()
c.add_movie()
a.current_status()
b.current_status()
print(f'총 영화수: {MovieTheater.total_movies}')
c.description()






