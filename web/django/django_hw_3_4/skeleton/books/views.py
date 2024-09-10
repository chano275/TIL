from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from . serializers import GenreSerializer
from . models import Genre



@api_view(['GET'])
def genre_detail(request, genre_pk):
    genre = Genre.objects.get(pk = genre_pk)
    serializer = GenreSerializer(genre)
    return Response(serializer.data)