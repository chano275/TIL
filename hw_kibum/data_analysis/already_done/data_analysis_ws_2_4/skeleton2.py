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


