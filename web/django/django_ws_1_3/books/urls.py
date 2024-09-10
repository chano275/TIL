from django.urls import path
from . import views

urlpatterns = [
    path('recommend/', views.recommend_books),
    path('<int:page>/', views.five_books),
]