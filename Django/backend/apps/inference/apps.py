import logging
import os
import sys
import threading

from django.apps import AppConfig
from django.conf import settings

logger = logging.getLogger(__name__)


class InferenceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.inference'

    def ready(self):
        """Optionally warm models at startup so first request is fast."""
        if not getattr(settings, 'ML_WARMUP_ON_STARTUP', False):
            return

        # Avoid duplicate warmup when Django autoreloader spawns parent process.
        if settings.DEBUG and os.environ.get('RUN_MAIN') != 'true':
            return

        # Skip warmup for commands where loading models is unnecessary.
        skip_commands = {'makemigrations', 'migrate', 'collectstatic', 'shell', 'test'}
        if any(command in sys.argv for command in skip_commands):
            return

        def _run_warmup():
            from ml_pipeline.model_loader import ModelLoader

            loader = ModelLoader()
            warmup_models = getattr(settings, 'ML_WARMUP_MODELS', None)
            statuses = loader.warmup_models(warmup_models)
            failed = [name for name, status in statuses.items() if not status.get('ok')]
            if failed:
                logger.warning('Model warmup completed with failures: %s', failed)
            else:
                logger.info('Model warmup completed for all configured models.')

        if getattr(settings, 'ML_WARMUP_ASYNC', True):
            thread = threading.Thread(target=_run_warmup, daemon=True)
            thread.start()
        else:
            _run_warmup()
