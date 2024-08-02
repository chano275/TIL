from rest_framework import serializers
from . models import Patient


class PatientListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        # exclude = ('created_at', 'updated_at',)


class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = '__all__'
        # # fields = ('id', 'title',)
        # exclude = ('created_at', 'updated_at',)


