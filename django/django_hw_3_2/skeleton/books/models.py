from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=100)
    author = models.ForeignKey("authors.Author", on_delete=models.CASCADE)

    def __str__(self):
        return self.title
    
    """
    Book 모델을 정의 > Author(저자)와 N:1 관계
    Author 및 Book의 데이터는 Admin site 를 통해 생성
    특정 저자의 이름과 해당 저자가 집필한 책의 개수를 반환하는 API 엔드포인트 작성
    저자의 pk를 기준으로 저자 정보를 반환 > 적절한 app에서 엔드포인트가 작성되어야
    """