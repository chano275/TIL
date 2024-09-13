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


