from django.db import models


class Patient(models.Model):
    first_name = models.CharField(max_length=30) # 
    last_name = models.CharField(max_length=30)
    birth_date = models.DateField()
    email = models.TextField()
    phone_number = models.CharField(max_length=15)
    created_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title
    
# Create your models here.
