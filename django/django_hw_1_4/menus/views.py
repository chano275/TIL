from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view

m_list = [
    {"name": "Espresso", "price": 3000},
    {"name": "Americano", "price": 3500},
    {"name": "Latte", "price": 4000}
]

@api_view(['GET'])
def menu_list(request):
    if request.method == "GET":
        return JsonResponse(m_list, safe=False)
"""
JsonResponse는 기본적으로 딕셔너리만 직렬화할 수 있으며, 리스트를 직렬화하려면 safe 매개변수를 False로
"""