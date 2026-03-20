"""
URL configuration for inference app.
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import BotDetectionView, BotDetectionResultViewSet, ModelReadinessView

router = DefaultRouter()
router.register(r'results', BotDetectionResultViewSet, basename='bot-detection-result')

urlpatterns = [
    path('detect/', BotDetectionView.as_view(), name='bot-detection'),
    path('health/models/', ModelReadinessView.as_view(), name='model-readiness'),
    path('', include(router.urls)),
]
