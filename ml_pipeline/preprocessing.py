"""
Text preprocessing utilities for ML pipeline.
"""
import re
from typing import List


class TextPreprocessor:
    """Handle text preprocessing for model input."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.max_length = 512
    
    def clean_text(self, text):
        """
        Clean and normalize text.
        
        Args:
            text (str): Raw input text
        
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters (optional - modify based on needs)
        # text = re.sub(r'[^a-zA-Z0-9\s.]', '', text)
        
        return text
    
    def tokenize(self, text):
        """
        Simple tokenization (in practice, use transformers tokenizer).
        
        Args:
            text (str): Input text
        
        Returns:
            List[str]: List of tokens
        """
        return text.split()
    
    def truncate(self, text):
        """
        Truncate text to max length.
        
        Args:
            text (str): Input text
        
        Returns:
            str: Truncated text
        """
        tokens = self.tokenize(text)
        truncated = tokens[:self.max_length]
        return ' '.join(truncated)
    
    def preprocess(self, text):
        """
        Full preprocessing pipeline.
        
        Args:
            text (str): Raw input text
        
        Returns:
            str: Preprocessed text
        """
        text = self.clean_text(text)
        text = self.truncate(text)
        return text
