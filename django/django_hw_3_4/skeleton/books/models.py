from django.db import models
from authors.models import Author

# Create your models here.
class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.ForeignKey("authors.Author", on_delete=models.CASCADE)
    # 아래와 같은 사왕에서 역참조 명 만들 필요? 원칙상은 X Genre.book_set.all() 치면 찾을 수 있음 
    # 안바꿔도 상관 없지만 장르 입장에서 book에 접근하는 이유가 m:n 관계라는 것을 명시하고자 한다면, related_name 정의해 주는게 .. 
    genres = models.ManyToManyField("books.Genre", related_name="books") # Genre 에 적어도 상관 없다. 
    
    def __str__(self):
        return self.title    

class Genre(models.Model):
    name = models.CharField(max_length=100)
    def __str__(self):
        return self.name
