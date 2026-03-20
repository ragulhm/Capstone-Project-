"""
URL configuration for bot_detection_system project.
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('apps.inference.urls_web')),
    path('api/', include('apps.inference.urls')),
]
