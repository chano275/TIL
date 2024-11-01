from rest_framework.decorators import api_view
from rest_framework.response import Response

from django.shortcuts import get_object_or_404
# 과제 4번 참조... 

from . serializers import AuthorSerializer
from . models import Author


@api_view(['GET'])
def author_detail(request, author_id):
    author = get_object_or_404(Author, pk=author_id)
    serializer = AuthorSerializer(author)
    return Response(serializer.data)