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
