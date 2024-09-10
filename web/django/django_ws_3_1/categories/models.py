from django.db import models

# name 필드 
class Category(models.Model):
    name = models.CharField(max_length = 100)
    def __str__(self):
        return self.name