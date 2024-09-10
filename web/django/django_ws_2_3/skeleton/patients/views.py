from .models import Patient
from .serializers import PatientSerializer, PatientListSerializer
from rest_framework.response import Response
from rest_framework.decorators import api_view

# Create your views here.
@api_view(["POST"])
def patient_create(request):
    serializer = PatientSerializer(data=request.data)
    if serializer.is_valid(raise_exception=True):
        serializer.save()
        return Response(serializer.data, status=201)
    

@api_view(['GET'])
def patient_list(request):
    patients = Patient.objects.all()
    serializer = PatientListSerializer(patients, many=True)
    return Response(serializer.data)
