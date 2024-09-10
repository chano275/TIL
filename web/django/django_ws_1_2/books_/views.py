from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from books import books as b_list
import random

@api_view(['GET'])
def recommend(request):
    best_sellers = []

    for book in b_list:
        if book['rating'] >= 6:
            best_sellers.append(book)

    return JsonResponse(random.choice(best_sellers))
