"""
Views for inference app.
"""
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Count
from django.shortcuts import render
from django.views import View
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import BotDetectionResult
from .serializers import TextInputSerializer, BotDetectionResultSerializer
from .services import InferenceService, MODEL_LOADER


MODEL_CHOICES = [
    'bert_fox',
    'roberta_fox',
    'distilbert_fox',
    'xlm_roberta_fox',
]


class BotDetectionView(APIView):
    """API endpoint for bot detection."""
    
    def post(self, request):
        """Process text and return bot detection result."""
        serializer = TextInputSerializer(data=request.data)
        
        if serializer.is_valid():
            text = serializer.validated_data['text']
            model = serializer.validated_data.get('model', 'bert_fox')
            
            try:
                inference_service = InferenceService()
                prediction, is_bot = inference_service.predict(text, model)
                
                # Save result to database
                result = BotDetectionResult.objects.create(
                    text=text,
                    model_used=model,
                    prediction=prediction,
                    is_bot=is_bot
                )
                
                return Response({
                    'text': text,
                    'model': model,
                    'prediction': prediction,
                    'is_bot': is_bot,
                    'id': result.id
                }, status=status.HTTP_200_OK)
            
            except Exception as e:
                return Response(
                    {'error': str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class BotDetectionResultViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for retrieving bot detection results."""
    queryset = BotDetectionResult.objects.all()
    serializer_class = BotDetectionResultSerializer


class ModelReadinessView(APIView):
    """Health endpoint for local model and tokenizer readiness."""

    def get(self, request):
        report = MODEL_LOADER.readiness_report()
        all_ready = all(
            item['model_ready'] and item['tokenizer_ready']
            for item in report.values()
        )
        return Response(
            {
                'status': 'ok' if all_ready else 'degraded',
                'offline_mode': True,
                'models': report,
            },
            status=status.HTTP_200_OK if all_ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class HomePageView(View):
    """Render home page and process inference form submissions."""

    template_name = 'inference/pages/home.html'

    def get(self, request):
        return render(request, self.template_name, self._build_context())

    def post(self, request):
        input_text = request.POST.get('text', '').strip()
        model = request.POST.get('model', MODEL_CHOICES[0])
        context = self._build_context(
            input_text=input_text,
            selected_model=model,
        )

        if not input_text:
            context['error'] = 'Please enter some text to analyze.'
            return render(request, self.template_name, context)

        if model not in MODEL_CHOICES:
            context['error'] = 'Invalid model selection.'
            return render(request, self.template_name, context)

        try:
            inference_service = InferenceService()
            prediction, is_bot = inference_service.predict(input_text, model)

            result = BotDetectionResult.objects.create(
                text=input_text,
                model_used=model,
                prediction=prediction,
                is_bot=is_bot,
            )

            context.update(
                {
                    'result': {
                        'id': result.id,
                        'model': model,
                        'prediction': prediction,
                        'is_bot': is_bot,
                    }
                }
            )
            context['history'] = self._history_queryset()
        except Exception as exc:
            context['error'] = f'Error while running inference: {exc}'

        return render(request, self.template_name, context)

    def _history_queryset(self):
        return BotDetectionResult.objects.order_by('-created_at')[:10]

    def _build_context(self, input_text='', selected_model='bert_fox'):
        return {
            'page_title': 'Home',
            'model_choices': MODEL_CHOICES,
            'selected_model': selected_model,
            'input_text': input_text,
            'history': self._history_queryset(),
        }


class DashboardPageView(LoginRequiredMixin, View):
    """Render dashboard page with aggregate metrics."""

    template_name = 'inference/pages/dashboard.html'

    def get(self, request):
        total = BotDetectionResult.objects.count()
        bot_count = BotDetectionResult.objects.filter(is_bot=True).count()
        human_count = total - bot_count
        bot_ratio = (bot_count / total * 100) if total else 0

        model_distribution = (
            BotDetectionResult.objects.values('model_used')
            .annotate(total=Count('id'))
            .order_by('-total')
        )

        context = {
            'page_title': 'Dashboard',
            'total_requests': total,
            'bot_count': bot_count,
            'human_count': human_count,
            'bot_ratio': bot_ratio,
            'model_distribution': model_distribution,
            'recent_results': BotDetectionResult.objects.order_by('-created_at')[:5],
        }
        return render(request, self.template_name, context)
