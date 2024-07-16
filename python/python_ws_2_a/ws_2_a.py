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