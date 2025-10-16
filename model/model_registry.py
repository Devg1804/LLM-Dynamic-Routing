"""
Contains the Model Registry, which serves as the definitive source of truth
for all available LLM models, their specifications, and performance profiles.
"""
import json

class ModelRegistry:
    """
    A class to load and provide easy access to the model specification data.
    """
    _MODEL_DATA = {
      "gemini-2.5-pro": {
        "model_name": "Gemini 2.5 Pro",
        "quality_score": 9.5,
        "latency_tier": "high", # high, medium, low, very_low
        "cost_per_million_tokens": { "input": 1.25, "output": 10.00 }
      },
      "gemini-2.5-flash": {
        "model_name": "Gemini 2.5 Flash",
        "quality_score": 8.8,
        "latency_tier": "low",
        "cost_per_million_tokens": { "input": 0.30, "output": 2.50 }
      },
      "gemini-2.5-flash-lite": {
        "model_name": "Gemini 2.5 Flash-Lite",
        "quality_score": 8.2,
        "latency_tier": "very_low",
        "cost_per_million_tokens": { "input": 0.10, "output": 0.40 }
      }
    }

    def get_all_models(self):
        """Returns a dictionary of all model specifications."""
        return self._MODEL_DATA

    def get_model(self, model_id: str):
        """
        Retrieves the specifications for a single model.
        
        Args:
            model_id (str): The ID of the model (e.g., 'gemini-2.5-pro').
            
        Returns:
            A dictionary of the model's specs, or None if not found.
        """
        return self._MODEL_DATA.get(model_id)


