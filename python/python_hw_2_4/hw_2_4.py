# 아래에 코드를 작성하시오.
class Animal():
    def __init__(self, name):
        self.name = name
    def speak(self):
        return 'Hello'

class Dog(Animal):
    def speak(self):
        return 'Woof!'
class Cat(Animal):
    def speak(self):
        return 'Meow!'

class Flyer():
    def fly(self):
        return "Flying"
class Swimmer():
    def swim(self):
        return "Swimming"
    
class Duck(Animal, Flyer, Swimmer):         
    def speak(self):
        return 'Quack!'

def make_animal_speak(str):
    print(str.speak())


a = Dog('d')
b = Cat('c')
c = Duck('q')

make_animal_speak(a)
make_animal_speak(b)
make_animal_speak(c)
print(c.fly())
print(c.swim())