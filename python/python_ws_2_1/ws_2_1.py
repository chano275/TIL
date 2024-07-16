# 아래에 코드를 작성하시오.
class MovieTheater():
    def __init__(self, name, total_seats):
        self.name = name
        self.total_seats = total_seats
        self.reserved_seats = 0
############################################################
    def __str__(self):          # __str__ : 객체 자체를 출력할 때 넘겨주는 형식을 지정해주는 메서드
        return self.name        #           return을 print 와 같이 이용 가능 / f'' 도 이용 가능 
############################################################
    
a = MovieTheater('메가박스', 150)
b = MovieTheater('CGV', 200)
print(a)
print(b)
