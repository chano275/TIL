from django.shortcuts import render
from . models import Patient
from . serializers import PatientSerializer, PatientListSerializer
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

# Create your views here.
@api_view(['GET', 'POST'])
def patient_list(request):
    if request.method == 'GET':
        patients = Patient.objects.all() 
        serializer = PatientSerializer(patients, many=True) 
        return Response(serializer.data)
    
    elif request.method == 'POST': # 새로운 article을 생성
        serializer = PatientSerializer(data=request.data)  
        if serializer.is_valid(): 
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)