
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .serializers import AuthorSerializer
from .models import Author

@api_view(['GET'])
def author_detail(request, author_pk):
    author = get_object_or_404(Author, pk=author_pk)
    serializer = AuthorSerializer(author)
    return Response(serializer.data)