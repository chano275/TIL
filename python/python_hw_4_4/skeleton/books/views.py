from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from . models import Book
from . serializers import BookSerializer

@api_view(['GET'])
@permission_classes([AllowAny])
def book_list(request):
    books = Book.objects.all()
    serializer = BookSerializer(books, many = True)
    return Response(serializer.data)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def book_borrow(request, book_isbn):
    book = Book.objects.get(isbn = book_isbn)
    book.borrowed = True
    book.save()

    serializer = BookSerializer(book)
    return Response(serializer.data)