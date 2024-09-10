from django.db import models

# 고객 - 상품 모델 many to one 관계 
# admin site에 생성한 모델 등록 / dummy data 생성 

class Product(models.Model):
    category = models.ForeignKey("categories.Category", on_delete=models.CASCADE)
    title = models.CharField(max_length = 100)
    description = models.CharField(max_length = 100)
    def __str__(self):
        return self.title