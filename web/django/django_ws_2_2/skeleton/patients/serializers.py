from .models import Patient
from rest_framework import serializers

class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        # fields = '__all__'
        fields = ('first_name', 'last_name', 'birth_date')