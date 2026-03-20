"""Model loader for local/offline fine-tuned transformer models."""

import logging
import os
from pathlib import Path
from threading import Lock

import torch
import torch.nn as nn
from django.conf import settings
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)


class BotDetectionModel(nn.Module):
    """Classifier head used during local .pt fallback loading."""

    def __init__(self, encoder: nn.Module, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
        nn.Linear(hidden_size, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_embedding)


class ModelLoader:
    """Load and cache models/tokenizers from local files only."""

    _MODEL_CACHE = {}
    _TOKENIZER_CACHE = {}
    _CACHE_LOCK = Lock()

    MODEL_SPECS = {
        'bert': {
            'hf_dir': 'bert_model',
            'pt_file': 'bert_base_10epoch.pth',
            'tokenizer_dir': 'bert_model',
            'base_model_name': 'bert-base-uncased',
            'hidden_size': 768,
        },
        'roberta': {
            'hf_dir': 'roberta_model',
            'pt_file': 'roberta_10epoch.pth',
            'tokenizer_dir': 'roberta_model',
            'base_model_name': 'roberta-base',
            'hidden_size': 768,
        },
        'distilbert': {
            'hf_dir': 'distilbert_model',
            'pt_file': 'distilbert_10epoch.pth',
            'tokenizer_dir': 'distilbert_model',
            'base_model_name': 'distilbert-base-uncased',
            'hidden_size': 768,
        },
        'xlm_roberta': {
            'hf_dir': 'xlm_roberta_model',
            'pt_file': 'xlm_roberta_10epoch.pth',
            'tokenizer_dir': 'xlm_roberta_model',
            'base_model_name': 'xlm-roberta-base',
            'hidden_size': 768,
        },
    }

    def __init__(self):
        self.models_dir = Path(getattr(settings, 'ML_MODELS_DIR', Path(__file__).resolve().parent.parent / 'ml_models'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Force transformers offline mode for production consistency.
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        os.environ.setdefault('HF_HUB_OFFLINE', '1')

    def normalize_model_name(self, model_name: str) -> str:
        normalized = (model_name or '').lower().strip().replace('-', '_').replace('_fox', '')
        if normalized not in self.MODEL_SPECS:
            raise ValueError(
                f"Unknown model '{model_name}'. Supported models: {', '.join(sorted(self.MODEL_SPECS.keys()))}"
            )
        return normalized

    def load_bundle(self, model_name: str):
        """Return a cached (model, tokenizer) tuple for the requested model."""
        normalized = self.normalize_model_name(model_name)

        with self._CACHE_LOCK:
            cached_model = self._MODEL_CACHE.get(normalized)
            cached_tokenizer = self._TOKENIZER_CACHE.get(normalized)
            if cached_model is not None and cached_tokenizer is not None:
                return cached_model, cached_tokenizer

        model = self.load_model(normalized)
        tokenizer = self.load_tokenizer(normalized)

        with self._CACHE_LOCK:
            self._MODEL_CACHE[normalized] = model
            self._TOKENIZER_CACHE[normalized] = tokenizer

        return model, tokenizer

    def get_model_source(self, model_name: str) -> str:
        """Return where model weights are expected to load from."""
        normalized = self.normalize_model_name(model_name)
        spec = self.MODEL_SPECS[normalized]
        hf_dir = self.models_dir / spec['hf_dir']
        pt_path = self.models_dir / spec['pt_file']

        if self._has_hf_model_files(hf_dir):
            return 'hf_local'
        if pt_path.exists():
            return 'pt_state_dict'
        return 'missing'

    def readiness_report(self):
        """Return per-model readiness and cache status for health checks."""
        report = {}
        for name in sorted(self.MODEL_SPECS.keys()):
            spec = self.MODEL_SPECS[name]
            source = self.get_model_source(name)
            tokenizer_path = self.models_dir / spec['tokenizer_dir']
            tokenizer_ready = self._has_tokenizer_files(tokenizer_path)
            with self._CACHE_LOCK:
                cached_model = name in self._MODEL_CACHE
                cached_tokenizer = name in self._TOKENIZER_CACHE

            report[name] = {
                'source': source,
                'model_ready': source in {'hf_local', 'pt_state_dict'},
                'tokenizer_ready': tokenizer_ready,
                'cached_model': cached_model,
                'cached_tokenizer': cached_tokenizer,
                'device': str(self.device),
            }
        return report

    def warmup_models(self, model_names=None):
        """Preload selected model/tokenizer bundles into cache."""
        names = model_names or sorted(self.MODEL_SPECS.keys())
        statuses = {}

        for name in names:
            normalized = self.normalize_model_name(name)
            try:
                self.load_bundle(normalized)
                statuses[normalized] = {'ok': True}
                logger.info('Warmup successful for model: %s', normalized)
            except Exception as exc:
                statuses[normalized] = {'ok': False, 'error': str(exc)}
                logger.exception('Warmup failed for model: %s', normalized)
        return statuses

    def load_model(self, model_name: str):
        """Load model from local HF directory first, then local .pt file."""
        normalized = self.normalize_model_name(model_name)

        with self._CACHE_LOCK:
            if normalized in self._MODEL_CACHE:
                return self._MODEL_CACHE[normalized]

        spec = self.MODEL_SPECS[normalized]
        hf_dir = self.models_dir / spec['hf_dir']
        pt_path = self.models_dir / spec['pt_file']

        try:
            if self._has_hf_model_files(hf_dir):
                logger.info('Loading Hugging Face model from local path: %s', hf_dir)
                model = AutoModelForSequenceClassification.from_pretrained(
                    hf_dir,
                    local_files_only=True,
                )
            elif pt_path.exists():
                logger.info('HF folder not found. Falling back to local .pt: %s', pt_path)
                model = self._load_pt_model(pt_path, normalized)
            else:
                raise FileNotFoundError(
                    f"No local model found for '{normalized}'. Expected either "
                    f"HF folder '{hf_dir}' or checkpoint '{pt_path}'."
                )

            model.to(self.device)
            model.eval()

            with self._CACHE_LOCK:
                self._MODEL_CACHE[normalized] = model

            return model
        except Exception as exc:
            logger.exception('Model loading failed for %s', normalized)
            raise RuntimeError(f"Failed to load model '{normalized}': {exc}") from exc

    def load_tokenizer(self, model_name: str):
        """Load tokenizer from local files only (offline-safe)."""
        normalized = self.normalize_model_name(model_name)

        with self._CACHE_LOCK:
            if normalized in self._TOKENIZER_CACHE:
                return self._TOKENIZER_CACHE[normalized]

        spec = self.MODEL_SPECS[normalized]
        tokenizer_candidates = [
            self.models_dir / spec['hf_dir'],
            self.models_dir / spec['tokenizer_dir'],
        ]

        for candidate in tokenizer_candidates:
            if self._has_tokenizer_files(candidate):
                try:
                    logger.info('Loading tokenizer from local path: %s', candidate)
                    tokenizer = AutoTokenizer.from_pretrained(candidate, local_files_only=True)
                    with self._CACHE_LOCK:
                        self._TOKENIZER_CACHE[normalized] = tokenizer
                    return tokenizer
                except Exception as exc:
                    logger.warning('Tokenizer load failed from %s: %s', candidate, exc)

        expected = ', '.join(str(path) for path in tokenizer_candidates)
        raise FileNotFoundError(
            f"Tokenizer files are missing for '{normalized}'. Provide local tokenizer files in: {expected}"
        )

    def _load_pt_model(self, checkpoint_path: Path, normalized_name: str):
        """Load .pt/.pth checkpoints while keeping architecture compatible."""
        spec = self.MODEL_SPECS[normalized_name]
        encoder = self._build_local_encoder(normalized_name)
        model = BotDetectionModel(encoder=encoder, hidden_size=spec['hidden_size'])

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = self._extract_state_dict(checkpoint)

        # Keep strict=False to tolerate minor key differences between training and serving.
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(
                'Model %s loaded with missing keys (strict=False): %s',
                normalized_name,
                missing_keys,
            )
        if unexpected_keys:
            logger.warning(
                'Model %s loaded with unexpected keys (strict=False): %s',
                normalized_name,
                unexpected_keys,
            )

        return model

    def _build_local_encoder(self, normalized_name: str):
        """Build encoder from local files only, never downloading from internet."""
        spec = self.MODEL_SPECS[normalized_name]
        local_encoder_dir = self.models_dir / spec['hf_dir']

        # Prefer local HF folder in ml_models/<model> when available.
        if self._has_hf_model_files(local_encoder_dir):
            return AutoModel.from_pretrained(local_encoder_dir, local_files_only=True)

        # Fallback to local cache using canonical pretrained model name.
        return AutoModel.from_pretrained(spec['base_model_name'], local_files_only=True)

    @staticmethod
    def _extract_state_dict(checkpoint_obj):
        """Support plain state_dict and common wrapped checkpoint formats."""
        if isinstance(checkpoint_obj, dict):
            if 'state_dict' in checkpoint_obj:
                return checkpoint_obj['state_dict']
            if 'model_state_dict' in checkpoint_obj:
                return checkpoint_obj['model_state_dict']
            if all(isinstance(k, str) for k in checkpoint_obj.keys()):
                return checkpoint_obj
        raise RuntimeError(
            'Unsupported checkpoint format. Expected a state_dict or a dict containing '
            "'state_dict'/'model_state_dict'."
        )

    @staticmethod
    def _has_hf_model_files(path: Path) -> bool:
        required = {'config.json'}
        if not path.exists() or not path.is_dir():
            return False
        names = {item.name for item in path.iterdir()}
        has_weight_file = any(name in names for name in ('pytorch_model.bin', 'model.safetensors'))
        return required.issubset(names) and has_weight_file

    @staticmethod
    def _has_tokenizer_files(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        names = {item.name for item in path.iterdir()}
        return any(name in names for name in ('tokenizer.json', 'vocab.txt', 'merges.txt', 'sentencepiece.bpe.model'))