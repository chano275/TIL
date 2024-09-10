from rest_framework import serializers
from books.serializers import BookSerializer
from .models import Author

class AuthorSerializer(serializers.ModelSerializer):
    book_count = serializers.IntegerField(source='book_set.count', read_only=True)
    # count << 수업에서 사용한 integerfield 사용 

    class Meta:
        model = Author
        fields = '__all__'