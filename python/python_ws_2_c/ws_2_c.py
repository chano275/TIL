# 아래에 코드를 작성하시오.
class User():
    def __init__(self, name, email):
        self.name = name
        self.email = email
    
    def show_info(self):
        print(f'Name: {self.name}, Email: {self.email}') # 사용자의 정보 출력 


class AdminUser(User):
    def __init__(self, name, email):
        super().__init__(name, email)
        self.permissions = 'Full Access'

    def show_info(self):
        print(f'Name: {self.name}, Email: {self.email}, Permissions: {self.permissions}') # 사용자의 정보 출력 
    

a = User('chano', '1@gmail.com')
b = User('jaeho', '2@gmail.com')
c = AdminUser('minsu', '3@gmail.com')

a.show_info()
b.show_info()
c.show_info()
