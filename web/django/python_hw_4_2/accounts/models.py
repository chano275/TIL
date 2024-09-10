from django.db import models
from django.contrib.auth.models import AbstractUser


# 추가
# accounts 앱 내에서 User 모델을 Django의 AbstractUser를 상속받아 정의
class User(AbstractUser):
    pass
