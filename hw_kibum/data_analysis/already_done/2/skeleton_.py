# 대기질 데이터를 초기화합니다.
temperature = 22.5
humidity = 60
pollutants = ["NO2", "O3", "PM2.5"]
location = {'latitude': 37.7749, 'longitude': -122.4194}
sensor_readings = (10, 20, 30)
status = "Operational"

# 문제 1: 각 변수의 타입 출력
# type 함수를 사용해 변수의 타입을 알아내고 print 함수로 출력
print(type(temperature), type(humidity), type(pollutants), type(location), type(sensor_readings), type(status))

# 문제 2: 리스트 pollutants에 대한 작업
print("문제 2. 리스트 pollutants에 대한 작업")
pollutants.append("CO2")# 1. 마지막에 "CO2"를 추가
pollutants.pop(0)       # 2. 첫 번째 요소 삭제
print("리스트 pollutants 대한 작업 후 pollutants의 길이:")
print(len(pollutants))  # 3. 리스트의 길이 출력
print("작업 후 pollutants 리스트의 요소:")# 리스트 pollutants에 대한 작업이 잘 수행되었는지 출력
print(pollutants)


# 문제 3: 튜플 sensor_readings에서 요소 20의 인덱스 찾기 및 출력
# index 메서드를 사용하여 요소 20의 인덱스를 찾아내 출력
print("문제 3. 튜플 sensor_readings에서 요소 20의 인덱스 찾기 및 출력")
print("튜플 sensor_readings에서 요소 20의 인덱스:")
for i in range(len(sensor_readings)):
    if sensor_readings[i] == 20:print(i)


# 문제 1: SensorData 클래스 정의 및 사용

class SensorData:
    def __init__(self, location, pollutant, value):      # 생성자를 사용해 클래스 SensorData의 속성을 초기화합니다.
        self.location = location
        self.pollutant = pollutant
        self.value = value
        pass
    def print_info(self):    # 센서 데이터를 출력하는 메서드 정의
        print(f"위치: {self.location}, 오염물질: {self.pollutant}, 농도: {self.value} μg/m³")

# 객체 생성 및 메서드 호출
sensor1 = SensorData("seoul", "PM3", 34.5)
sensor2 = SensorData("daejeon", "PM2.1", 35.5)

print("문제 1. SensorData 클래스 정의 및 사용")
print("객체 sensor1의 정보(위치, 오염물질, 농도)")
sensor1.print_info()# print_info 메서드를 사용해 객체 sensor1의 정보를 출력합니다.

print("\n객체 sensor2의 정보(위치, 오염물질, 농도)")
sensor2.print_info()# print_info 메서드를 사용해 객체 sensor2의 정보를 출력합니다.



# 문제 2: alarm 메서드 추가

class SensorData:
    def __init__(self, location, pollutant, value):
        self.location = location
        self.pollutant = pollutant
        self.value = value

    def print_info(self):
        print(f"위치: {self.location}, 오염물질: {self.pollutant}, 농도: {self.value} μg/m³")

    # 임계값과 비교하여 경고 메시지 출력하는 메서드 정의
    def alarm(self, threshold):
        if self.value > threshold:
            print('경고')


# 객체 생성
sensor1 = SensorData("Downtown", "PM2.5", 35.4)

print("문제 2. alarm 메서드 추가")
print("alram 메서드를 통해 측정값과 임계값을 비교한 결과")
sensor1.alarm(25.0)  # alarm 메서드를 사용해 객체 sensor1의 경고 메시지를 출력합니다.

# 문제 3: update_value 메서드 추가

class SensorData:
    def __init__(self, location, pollutant, value):
        self.location = location
        self.pollutant = pollutant
        self.value = value

    def print_info(self):
        print(f"위치: {self.location}, 오염물질: {self.pollutant}, 농도: {self.value} μg/m³")

    def alarm(self, threshold):
        if self.value > threshold:
            print(f"경고! {self.location}의 {self.pollutant} 농도가 임계값을 초과했습니다!")
        # 임계값을 초과하지 않을 경우 안전 메시지 출력
        else:
            print(f"{self.location}의 {self.pollutant} 농도가 안전 범위 내에 있습니다.")

    # 새로운 측정값을 입력받아 value 속성을 업데이트하고 업데이트 메시지를 출력하는 메서드 정의
    def update_value(self, new_val):
        self.value = new_val

# 객체 생성
sensor1 = SensorData("Downtown", "PM2.5", 35.4)
sensor1.print_info()

print("문제 3. update_value 메서드 추가\n")
print("update_value 메서드를 통해 측정값을 업데이트한 결과")
sensor1.update_value(40.0)# update_value 메서드를 사용해 객체 sensor1의 측정값을 업데이트합니다.
print("\n객체 sensor1의 정보(위치, 오염물질, 농도)")
sensor1.print_info()# print_info 메서드를 사용해 객체 sensor1의 정보를 출력합니다.

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
# 문제 1: 기본 클래스와 상속 클래스 정의
class PollutionSource:# PollutionSource 클래스를 정의
    def __init__(self, name, emission):                                    # 생성자: name, emission를 매개변수로 받음
        self.name = name
        self.emission = emission

    def describe(self):                                                    # describe 메서드 정의: 객체의 정보를 문자열로 반환
        return f"오염원: {self.name}, 배출량: {self.emission} 톤/년"        # 오염원, 배출량을 문자열로 반환

class Traffic(PollutionSource):# Traffic 클래스를 정의 / # PollutionSource 클래스를 상속
    def __init__(self, name, emission):
        super().__init__(name, emission)

    def describe(self):                                                    # describe 메서드 오버라이딩
        return f"오염원: {self.name}, 배출량: {self.emission} 톤/년"        # 오염원, 배출량을 문자열로 반환

class Industry(PollutionSource):# Industry 클래스를 정의 / # PollutionSource 클래스를 상속
    def __init__(self, name, emission):
        super().__init__(name, emission)

    def describe(self):                                                    # describe 메서드 오버라이딩
        return f"오염원: {self.name}, 배출량: {self.emission} 톤/년"        # 오염원, 배출량을 문자열로 반환

class Natural(PollutionSource):# Natural 클래스를 정의 /  PollutionSource 클래스를 상속
    def __init__(self, name, emission):
        super().__init__(name, emission)

    def describe(self):                                                    # describe 메서드 오버라이딩
        return f"오염원: {self.name}, 배출량: {self.emission} 톤/년"        # 오염원, 배출량을 문자열로 반환

# Traffic, Industry, Natural 클래스의 객체 생성
traffic = Traffic('1', '2')
industry = Industry('3', '4')
natural = Natural('5', '6')

# 객체의 describe 메서드 호출해서 정보 출력
print("문제 1. 기본 클래스와 상속 클래스 정의")
print("traffic 객체 정보:", traffic.describe())
print("industry 객체 정보:", industry.describe())
print("natural 객체 정보:", natural.describe())


# 문제 2: move 메서드 오버라이딩

class PollutionSource:
    def __init__(self, name, emission):
        self.name = name
        self.emission = emission

    def describe(self):
        return f"오염원: {self.name}, 배출량: {self.emission} 톤/년"

    # 고유한 이동 방식을 반환하는 move 메서드 정의
    def move(self):
        return "오염원은 고정되어 있습니다."

class Traffic(PollutionSource):
    def __init__(self, name, emission):
        super().__init__(name, emission)

    def move(self):    # move 메서드 오버라이딩
        return f"{self.name}은 움직이고 있습니다."

class Industry(PollutionSource):
    def __init__(self, name, emission):
        super().__init__(name, emission)

    def move(self):    # move 메서드 오버라이딩
        return f"{self.name}은 움직이고 있습니다."
    
class Natural(PollutionSource):
    def __init__(self, name, emission):
        super().__init__(name, emission)

    def move(self):    # move 메서드 오버라이딩
        return f"{self.name}은 움직이고 있습니다."

# Traffic, Industry, Natural 클래스의 객체 생성
traffic = Traffic("자동차 배출", 50)
industry = Industry("공장 배출", 100)
natural = Natural("자연 배출", 200)

print("문제 2. move 메서드 오버라이딩")
print("traffic 객체의 이동 방식:", traffic.move())
print("industry 객체의 이동 방식:", industry.move())
print("natural 객체의 이동 방식:", natural.move())


# 문제 3: 상위 클래스 메서드 호출

class PollutionSource:
    def __init__(self, name, emission):
        self.name = name
        self.emission = emission

    def describe(self):
        return f"[부모 클래스] 오염원: {self.name}, 배출량: {self.emission} 톤/년"

    def move(self):
        return "오염원은 고정되어 있습니다."

class Traffic(PollutionSource):
    def __init__(self, name, emission):
        super().__init__(name, emission)
class Industry(PollutionSource):
    def __init__(self, name, emission):
        super().__init__(name, emission)
class Natural(PollutionSource):
    def __init__(self, name, emission):
        super().__init__(name, emission)

# Traffic, Industry, Natural 클래스의 객체 생성
traffic = Traffic("차량 배출", 50)
industry = Industry("공장 배출", 100)
natural = Natural("화산 폭발", 200)

print("문제 3. 상위 클래스 메서드 호출")
print("자식 클래스(Traffic)와 부모 클래스(PollutionSource)의 정보 출력")
print(traffic.describe())

print("\n자식 클래스(Industry)와 부모 클래스(PollutionSource)의 정보 출력")
print(industry.describe())

print("\n자식 클래스(Natural)와 부모 클래스(PollutionSource)의 정보 출력")
print(natural.describe())

# 문제 1: 단어 수 계산 후 파일에 저장

# 텍스트 파일의 내용을 읽어들이기 / open 함수를 사용하여 data/data.txt 파일을 읽기 모드("r")로 열고, file 변수에 저장
with open("C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_ws_2_5/data.txt", "r") as file:

# 각 줄의 단어 수 계산 / 반복문을 사용하여 lines의 각 요소를 line으로 하나씩 꺼냄.
# split 함수를 사용하여 단어로 분리한 후 len 함수를 사용하여 단어 수를 계산하여 word_counts 리스트에 저장
    lines = file.readlines()
    word_counts = [len(line.split()) for line in lines]

# 중간 결과 출력 / enumerate > word_counts의 각 요소와 인덱스를 i, count로 하나씩 꺼내서 출력
# f-string을 사용하여 i+1번째 줄: count개의 단어 형식으로 출력
for i, count in enumerate(word_counts):
    print(f'{i+1}번째 줄에 {count}개 단어')

# 결과를 파일에 저장
# open 함수를 사용하여 data/word_count.txt 파일을 쓰기 모드("w")로 열고, file 변수에 저장
# 반복문을 사용하여 word_counts의 각 요소와 인덱스를 i, count로 하나씩 꺼내서 파일에 쓰기
with open("C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_ws_2_5/word_count.txt", "w") as file:
    for i, count in enumerate(word_counts):
        file.write(f'{i+1}번째 줄에 {count}개 단어\n')


# 문제 2: 가장 긴 단어 찾기

# data/data.txt 파일을 읽기 모드("r")로 열고, file 변수에 저장


# 가장 긴 단어와 그 단어가 있는 줄 번호를 저장할 변수 초기화
longest_word = ""
longest_word_line = 0

# enumerate 함수를 사용하여 lines의 각 요소와 인덱스를 i, line으로 하나씩 꺼내서 반복문을 실행
# 만약 가장 긴 단어가 2개라면 먼저 발견된 단어가 선택됨


# 중간 결과 출력
# 가장 긴 단어와 그 단어가 있는 줄 번호 출력
print(f"가장 긴 단어: {longest_word}")
print(f"{longest_word_line}번째 줄에서 발견됨")

# 결과를 파일에 저장
# data/last_word.txt 파일에 가장 긴 단어와 그 단어가 있는 줄 번호를 저장
# open 함수를 사용하여 data/longest_word.txt 파일을 쓰기 모드("w")로 열고, file 변수에 저장


# 문제 3: 파일 내용을 거꾸로 저장

with open("C:/Users/SSAFY/Desktop/TIL/GitAuto/data_analysis/data_analysis_ws_2_5/data.txt", "r") as file:
    lines = file.readlines()

# 거꾸로 변환된 내용을 저장할 리스트 초기화
# 슬라이싱은 문자열의 일부분을 추출하는 방법으로, [시작 인덱스:끝 인덱스:간격] 형식으로 사용합니다.
# 시작 인덱스와 끝 인덱스를 생략하면 문자열의 처음부터 끝까지 추출하며, 간격을 음수로 지정하면 문자열을 거꾸로 만들 수 있습니다.
reversed_lines = [line[::-1] for line in lines[::-1]]

# 중간 결과 출력
print("거꾸로 변환된 내용:")
print(reversed_lines)



for r in reversed_lines:# reversed_lines의 각 요소를 line으로 하나씩 꺼내서 출력
    print(r, end = '')    # end="": print 함수가 줄바꿈을 하지 않도록 설정    # 이미 각 요소의 끝에 줄바꿈이 포함되어 있기 때문에 줄바꿈을 두 번 하지 않도록 설정



# 결과를 파일에 저장

