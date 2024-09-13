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

