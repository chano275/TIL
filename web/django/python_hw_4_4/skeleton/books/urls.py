from django.urls import path
from . import views

urlpatterns = [
    path('', views.book_list),
    path('<int:book_isbn>/', views.book_borrow),
]
