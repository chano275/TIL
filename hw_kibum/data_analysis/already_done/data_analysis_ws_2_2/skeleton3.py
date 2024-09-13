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

