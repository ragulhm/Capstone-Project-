"""
Serializers for inference app.
"""
from rest_framework import serializers
from .models import BotDetectionResult


class TextInputSerializer(serializers.Serializer):
    """Serializer for text input."""
    text = serializers.CharField(required=True)
    model = serializers.CharField(required=False, default='bert_fox')


class BotDetectionResultSerializer(serializers.ModelSerializer):
    """Serializer for bot detection results."""
    
    class Meta:
        model = BotDetectionResult
        fields = ['id', 'text', 'model_used', 'prediction', 'is_bot', 'created_at']
        read_only_fields = ['id', 'created_at']
