from rest_framework.decorators import api_view
from .models import Post
from .serializers import PostSerializer
from rest_framework.response import Response
from rest_framework import status


@api_view(['GET', 'POST'])
def post_list(request):
    if request.method == 'GET':
        posts = Post.objects.all()
        serializer = PostSerializer(posts, many=True) 
        return Response(serializer.data) 
    
    elif request.method == 'POST': 
        serializer = PostSerializer(data=request.data) 
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['DELETE', 'PUT'])
def post_detail(request, post_pk):
    post = Post.objects.get(pk=post_pk)
    
    if request.method == 'DELETE':
        post.delete()
        return Response({'message':'Post deleted successfully.'}, 
										    status = status.HTTP_204_NO_CONTENT)
    
    elif request.method == 'PUT':
        serializer = PostSerializer(post, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
