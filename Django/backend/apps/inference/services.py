"""
Service layer for inference operations.
"""
from ml_pipeline.model_loader import ModelLoader
from ml_pipeline.inference import Inference


MODEL_LOADER = ModelLoader()


class InferenceService:
    """Service class for handling model inference."""
    
    def __init__(self):
        """Initialize the inference service."""
        self.model_loader = MODEL_LOADER
        self.inference = Inference()
    
    def predict(self, text, model_name='bert_fox'):
        """
        Predict if text is bot-generated.
        
        Args:
            text (str): Input text to classify
            model_name (str): Name of the model to use
        
        Returns:
            tuple: (prediction_score, is_bot_bool)
        """
        try:
            # Load model/tokenizer from local cache or disk once.
            model, tokenizer = self.model_loader.load_bundle(model_name)
            
            # Preprocess text
            preprocessed_text = self.inference.preprocess(text)
            
            # Make prediction
            prediction = self.inference.infer(model, tokenizer, preprocessed_text)
            
            # Determine if bot (threshold = 0.5)
            is_bot = prediction > 0.5
            
            return float(prediction), bool(is_bot)
        
        except Exception as e:
            raise Exception(f"Inference failed: {str(e)}")
