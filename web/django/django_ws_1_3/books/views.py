from rest_framework.decorators import api_view
from django.http import JsonResponse
import random

books = [
    {
        "pk": 1,
        "title": "Python Programming for Beginners",
        "description": "An introductory guide to Python programming.",
        "published_date": "2020-05-15",
        "rating": 8.2
    },
    {
        "pk": 2,
        "title": "Advanced Python Techniques",
        "description": "Explore advanced features and techniques in Python.",
        "published_date": "2021-08-22",
        "rating": 9.0
    },
    {
        "pk": 3,
        "title": "Data Science with Python",
        "description": "Learn data science concepts and tools using Python.",
        "published_date": "2019-11-30",
        "rating": 5.5
    },
    {
        "pk": 4,
        "title": "Machine Learning with Python",
        "description": "A comprehensive guide to machine learning with Python.",
        "published_date": "2018-07-10",
        "rating": 7.2
    },
    {
        "pk": 5,
        "title": "Web Development with Django",
        "description": "Build powerful web applications using Django and Python.",
        "published_date": "2022-01-15",
        "rating": 8.1
    },
    {
        "pk": 6,
        "title": "Python for Data Analysis",
        "description": "Techniques and tools for data analysis with Python.",
        "published_date": "2017-03-20",
        "rating": 6.6
    },
    {
        "pk": 7,
        "title": "Automate the Boring Stuff with Python",
        "description": "Automate common tasks and improve productivity with Python.",
        "published_date": "2015-04-14",
        "rating": 9.4
    },
    {
        "pk": 8,
        "title": "Fluent Python",
        "description": "Write efficient, high-quality Python code.",
        "published_date": "2019-09-05",
        "rating": 5.5
    },
    {
        "pk": 9,
        "title": "Effective Python",
        "description": "59 specific ways to improve your Python skills.",
        "published_date": "2017-12-11",
        "rating": 7.7
    },
    {
        "pk": 10,
        "title": "Python Crash Course",
        "description": "A hands-on, project-based introduction to Python.",
        "published_date": "2016-06-17",
        "rating": 6.5
    }
]

# Create your views here.
@api_view(['GET'])
def recommend_books(request):
    high_rated_books = [book for book in books if book['rating'] >= 6.0]
    recommended_books = random.sample(high_rated_books, 1)
    return JsonResponse(recommended_books[0])

def five_books(request, page): 
    # APP 의 urls 에서 변수 받아오면, 
    # views 의 함수에서 매개변수에 넣어줘야 받아올 수 있음
    
    len_books = len(books)
    if not page: page == 1
    if page * 5 > len_books: return JsonResponse([], safe=False)
    return JsonResponse(books[(page-1) * 5 : (page) * 5], safe=False) # 0 ~ 4 / 5 ~ 9 ...  

