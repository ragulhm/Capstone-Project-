"""
Inference pipeline for bot detection.
"""
import torch
from ml_pipeline.preprocessing import TextPreprocessor


class Inference:
    """Handle inference operations."""
    
    def __init__(self):
        """Initialize the inference pipeline."""
        self.preprocessor = TextPreprocessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def preprocess(self, text):
        """
        Preprocess text for model input.
        
        Args:
            text (str): Input text
        
        Returns:
            str: Preprocessed text
        """
        return self.preprocessor.preprocess(text)
    
    def infer(self, model, tokenizer, text):
        """
        Run inference on preprocessed text.
        
        Args:
            model: Loaded model
            tokenizer: Loaded tokenizer
            text (str): Preprocessed text
        
        Returns:
            float: Prediction score (0-1)
        """
        try:
            # Tokenize text for transformer input.
            inputs = tokenizer(
                text,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = model(**inputs)

                # Support Hugging Face sequence classification models.
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits

                    if logits.shape[-1] == 1:
                        prediction = torch.sigmoid(logits)[0, 0].item()
                    else:
                        probs = torch.softmax(logits, dim=-1)
                        prediction = probs[0, 1].item()
                else:
                    # Custom fallback model returns raw logits; convert to probability.
                    raw_output = outputs[0] if isinstance(outputs, tuple) else outputs
                    prediction = torch.sigmoid(raw_output.squeeze()).item()
            
            return prediction
        
        except Exception as e:
            raise Exception(f"Inference error: {str(e)}")
