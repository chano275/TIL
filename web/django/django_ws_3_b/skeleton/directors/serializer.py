from rest_framework import serializers
from . models import Director
from movies.models import Movie

class DirectorListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Director
        exclude = ('created_at', 'updated_at',)


# 게시글 조회 할 때 해당 게시글의 댓글도 함께 조회
class DirectorSerializer(serializers.ModelSerializer):
    class MovieSerializer(serializers.ModelSerializer):
        class Meta:
            model = Movie
            fields = ('id', 'content',)   

    movie_set = MovieSerializer(many=True, read_only=True) 
    comment_count = serializers.IntegerField(source='movie_set.count', read_only=True)

    # other_field = serializers.SerializerMethodField()
    # def get_other_field(self, obj):
    #     return f'게시글 제목은 "{obj.title}" 이렇게 나타납니다.'
    
    class Meta:
        model = Director
        fields = '__all__'
        # # fields = ('id', 'title',)
        # exclude = ('created_at', 'updated_at',)