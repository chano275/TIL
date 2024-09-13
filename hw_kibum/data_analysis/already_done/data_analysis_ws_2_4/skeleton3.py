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