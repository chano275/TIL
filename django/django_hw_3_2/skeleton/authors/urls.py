from django.urls import path
from . import views

urlpatterns = [
    path('<int:author_id>/', views.author_detail), 
    # author 다음 pk 숫자대로 author 의 id에 따라서 출력되는걸 result에서 볼 수 있음
]