# articles/serializers.py

from rest_framework import serializers
from . models import Post

class PostListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Post  # Model: 직렬화할 대상 모델
        # exclude = ('created_at', 'updated_at',)

class PostSerializer(serializers.ModelSerializer):
    class Meta:
        model = Post
        fields = '__all__'
        # # fields = ('id', 'title',)
        # exclude = ('created_at', 'updated_at',)


