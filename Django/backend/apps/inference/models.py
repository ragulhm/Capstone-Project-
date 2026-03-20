"""
Models for inference app.
"""
from django.db import models


class BotDetectionResult(models.Model):
    """Store bot detection results."""
    
    text = models.TextField()
    model_used = models.CharField(max_length=50)
    prediction = models.FloatField()
    is_bot = models.BooleanField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Detection: {self.model_used} - {self.is_bot}"
