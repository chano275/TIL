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