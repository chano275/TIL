from django.contrib import admin
from django.urls import path, include

# my_api/urls.py

urlpatterns = [
    path('admin/', admin.site.urls),


    path('accounts/', include('dj_rest_auth.urls')),
    path('accounts/registration/', include('dj_rest_auth.registration.urls')),
    path('books/', include('books.urls')),
    
]
