# 모든 학생의 평균 점수를 계산하여 출력하시오.
# 80점 이상을 받은 학생들의 이름을 리스트 컴프리헨션을 사용하여 추출하시오.
# 학생들의 점수를 높은 순서대로 정렬하여 출력하시오.
# 점수가 가장 높은 학생과 가장 낮은 학생의 점수 차이를 계산하여 출력하시오.
# 각 학생의 점수가 평균 점수보다 높은지 낮은지를 판단하여, 낮은 학생의 이름과 성적을 함께 출력하시오.

# '=' 포커싱 후 Ctrl + D > 동일한 단어 찾고 변환도 가능
# 학생 점수 정보
A = {   "Alice" : 85,
   "Bob" : 78,
   "Charlie" : 92,
   "David" : 88,
   "Eve" : 95}

# 아래에 코드를 작성하시오.

student = {"Alice" : 85,   "Bob" : 78,   "Charlie" : 92,   "David" : 88,   "Eve" : 95}


print('1. 학생들의 이름과 점수를 딕셔너리에 저장')
print(f'students type: {type(student)}')
print(f'학생들의 이름과 점수: {student}')

py = sum(student.values()) / len(student)
print(f'2. 모든 학생의 평균 점수: {py:.2f}') ## 출력시 형태 

# 튜플로 만든 list < value가 80 이상이면 
# list comprehension ! 
# top_students = [(key, value) for key, value in student.items() if value >= 80]
# print(top_students)

ans = []
print('3. 기준 점수(80점) 이상을 받은 학생 수:', end = ' ')
for k,v in student.items():
    if v >= 80:
        ans.append(k)
print(ans)


print('4. 점수 순으로 정렬: ')
sorted_d = sorted(student.items(), key=lambda x: x[1], reverse=True)
# sorted : 원래 key 값 기준으로 정렬됨. 
#          첫번째 인자 : 반복 가능한 객체 / dict가 가지고 있는 키와 value 다 필요하므로 items 가져옴 
# key : 무엇을 기준으로 정렬 ? > value  >> 람다 사용법 ㄹㄹㄹ 


for q in sorted_d:
    print(f'{q[0]}: {q[1]}')

diff = max(student.values()) - min(student.values())
print(f'5. 점수가 가장 높은 학생과 가장 낮은 학생의 점수 차이: {diff}')

print('6. 각 학생의 점수가 평균보다 높은지 낮은지 판단: ')
for k,v in student.items():
    if v <= py:
        print(f'{k} 학생의 점수 ({v})는 평균 이하입니다.')
    else:
        print(f'{k} 학생의 점수 ({v})는 평균 이상입니다.')

########################################################################################################

# 아래에 코드를 작성하시오.
class Product():
    product_count = 0
    def __init__(self, name, price):
        self.name = name
        self.price = price
        Product.product_count += 1              
        # * 여기서 self를 불러버리면 전체적인 count를 클래스에서 체크 불가 
    def display_info(self):
        print(f'상품명: {self.name}, 가격: {self.price}원')

p1 = Product('사과', 1000)
p2 = Product('바나나', 1500)

p1.display_info()
p2.display_info()
print(f'총 상품 수: {Product.product_count}')

########################################################################################################

# 아래에 코드를 작성하시오.
class Animal():
    def __init__(self, name):
        self.name = name
    def speak(self):
        return 'Hello'

class Dog(Animal):
    def speak(self):
        return 'Woof!'
class Cat(Animal):
    def speak(self):
        return 'Meow!'

class Flyer():
    def fly(self):
        return "Flying"
class Swimmer():
    def swim(self):
        return "Swimming"
    
class Duck(Animal, Flyer, Swimmer):         
    def speak(self):
        return 'Quack!'

def make_animal_speak(str):
    print(str.speak())


a = Dog('d')
b = Cat('c')
c = Duck('q')

make_animal_speak(a)
make_animal_speak(b)
make_animal_speak(c)
print(c.fly())
print(c.swim())

########################################################################################################
ans = []
for i in range(len(movies)):
    dict = {}
    dict['title'] = movies[i]
    dict['rating'] = ratings[i]
    ans.append(dict)
print(ans)

# zip 써서 
# print("******")
# print(list(zip(movies, ratings)))
# for title, rating in zip(movies, ratings):
#     temp_dict = {'title':title, 'rating':rating}
#     list.append(temp_dict)

# print("******")

############################################

print('Enter the rating threshold:', end = ' ')
a = float(input())
li = (get_high_rated_movies(ans, a))
print(f'Movies with a rating of {(a)} or higher')
for j in range(len(li)):
    print(li[j])

########################################################################################################

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


########################################################################################################
# 아래에 코드를 작성하시오.
class MovieTheater():

    def __init__(self, name, total_seats):    # 생성자 내에 들어가는 인스턴스 변수 : 각 인스턴스별로 따로 관리 
        self.name = name
        self.total_seats = total_seats
        self.reserved_seats = 0

    def __str__(self):          # __str__ : 객체 자체를 출력할 때 넘겨주는 형식을 지정해주는 메서드
        return self.name        #           return을 print 와 같이 이용 가능 / f'' 도 이용 가능 

############################################################
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

a = MovieTheater('메가박스', 150)
b = MovieTheater('CGV', 200)

a.reserve_seat()
a.reserve_seat()
a.reserve_seat()
a.current_status()
b.reserve_seat()
b.current_status()

########################################################################################################
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


########################################################################################################
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








########################################################################################################
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

########################################################################################################
# 아래에 코드를 작성하시오.
class User():
    user_count = 0
    def __init__(self, name, email):
        User.user_count += 1
        self.name = name
        self.email = email
    
    @staticmethod
    def description():
        print('SNS 사용자는 소셜 네트워크 서비스를 이용하는 사람을 의미합니다.')


a = User('chano', '1@gmail.com')
b = User('jaeho', '2@gmail.com')

print(f'{a.name}')
print(f'{a.email}')
print(f'{b.name}')
print(f'{b.email}')
print(f'현재까지 생성된 사용자 수: {User.user_count}')
a.description()

########################################################################################################
# 아래에 코드를 작성하시오.
class Post():
    post_count = 0

    def __init__(self, title, content):
        self.title = title
        self.content = content
        Post.post_count += 1

    def show_content(self):
        print(self.content)
    
    def total_posts(self): # 게시물 총 개수 출력
        print(f'Total posts: {Post.post_count}')

    @staticmethod
    def description():
        print('SNS 게시물은 소셜 네트워크 서비스에서 사용자가 작성하는 글을 의미합니다.')

a = Post('제주도', '제주도 여행 왔어요~')
b = Post('일본', '일본 여행 왔어요~')


print(f'Title: {a.title}')
print(f'Content: {a.content}')
print(f'Title: {b.title}')
print(f'Content: {b.content}')
a.total_posts()
Post.description()

########################################################################################################
# 아래에 코드를 작성하시오.
class User():
    def __init__(self, name, email):
        self.name = name
        self.email = email
    
    def show_info(self):
        print(f'Name: {self.name}, Email: {self.email}') # 사용자의 정보 출력 


class AdminUser(User):
    def __init__(self, name, email):
        super().__init__(name, email)
        self.permissions = 'Full Access'

    def show_info(self):
        print(f'Name: {self.name}, Email: {self.email}, Permissions: {self.permissions}') # 사용자의 정보 출력 
    

a = User('chano', '1@gmail.com')
b = User('jaeho', '2@gmail.com')
c = AdminUser('minsu', '3@gmail.com')

a.show_info()
b.show_info()
c.show_info()

#######################################################################################################