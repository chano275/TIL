# 아래에 코드를 작성하시오.
class MovieTheater():
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
        print(f'총 좌석 수: {self.total_seats}')
        print(f'예약된 좌석 수: {self.reserved_seats}')


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
            print('예약 가능한 VIP 좌석이 없습니다.')   
            super().reserve_seat()

############################################################

a = MovieTheater('메가박스', 150)
b = MovieTheater('CGV', 200)
c = VIPMovieTheater('롯데시네마',300,3)

c.reserve_seat()
c.reserve_seat()
c.reserve_seat()
c.reserve_seat()
