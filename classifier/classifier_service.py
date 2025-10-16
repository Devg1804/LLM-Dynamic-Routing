"""
Provides a high-level, easy-to-use service to interact with the
NVIDIA Prompt Classifier. This is the intended entry point for other
parts of the application.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

import torch
from transformers import AutoConfig, AutoTokenizer

# Import the PyTorch model architecture from our other file
from classifier.classifier_model import CustomModel


@dataclass
class ClassificationResult:
    """A structured dataclass to hold the classifier's output."""
    task_type_1: str
    task_type_2: str
    task_type_prob: float
    creativity_scope: float
    reasoning: float
    contextual_knowledge: float
    number_of_few_shots: int
    domain_knowledge: float
    no_label_reason: float
    constraint_ct: float
    prompt_complexity_score: float


class NvidiaClassifier:
    """
    A wrapper for the NVIDIA classifier model that handles loading,
    tokenization, and inference, providing a clean interface.
    """
    MODEL_NAME = "nvidia/prompt-task-and-complexity-classifier"

    def __init__(self):
        """
        Initializes the classifier by loading the model, config, and tokenizer
        from Hugging Face. This might take a moment on first run.
        """
        print("Initializing classifier and loading model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        config = AutoConfig.from_pretrained(self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        
        # Instantiate our custom model class and load pre-trained weights
        self.model = CustomModel(
            target_sizes=config.target_sizes,
            task_type_map=config.task_type_map,
            weights_map=config.weights_map,
            divisor_map=config.divisor_map,
        ).from_pretrained(self.MODEL_NAME)
        
        self.model.to(self.device)
        self.model.eval()
        print("Classifier initialized successfully.")

    def classify(self, prompt: str) -> ClassificationResult:
        """
        Takes a single prompt string and returns a structured ClassificationResult.

        Args:
            prompt (str): The user prompt to classify.

        Returns:
            ClassificationResult: A dataclass object with the full analysis.
        """
        # The model expects a list of prompts, so we wrap the single prompt.
        results = self.classify_batch([prompt])
        return results[0] # Return the first (and only) result

    def classify_batch(self, prompts: List[str]) -> List[ClassificationResult]:
        """
        Takes a batch of prompt strings and returns a list of results.

        Args:
            prompts (List[str]): A list of user prompts to classify.

        Returns:
            List[ClassificationResult]: A list of dataclass objects.
        """
        encoded_texts = self.tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
        ).to(self.device)

        
        raw_results = self.model(encoded_texts)

        # The model returns results for the batch in a dictionary of lists.
        # We need to transpose this into a list of result objects.
        results_list = []
        num_prompts = len(prompts)
        for i in range(num_prompts):
            result_dict = {key: value[i] for key, value in raw_results.items()}
            results_list.append(ClassificationResult(**result_dict))
            
        return results_list

