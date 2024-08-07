from django.db import models
from django.conf import settings
# 도서 모델 
# borrowed 필드 : 현재 대여 상태 
# 단, POST 대여 요청 로직 작성시, 
# 현재 대여 되어 있는지에 대한 예외 처리는 
# 별도로 작성하지 않아도 무관하다.

class Genre(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    title = models.CharField(max_length=200)
    genres = models.ManyToManyField(Genre, related_name='books')
    published_date = models.DateField()
    borrowed = models.BooleanField(default=False)
    isbn = models.CharField(max_length=13, unique=True)


