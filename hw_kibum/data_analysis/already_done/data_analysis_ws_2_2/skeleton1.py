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



