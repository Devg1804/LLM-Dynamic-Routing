"""
The core decision-making component of the LLM Router.
It synthesizes signals from the classifier and semantic search, normalizes
them, and applies weights to select the optimal model.
"""
import json
from typing import List, Dict, Any
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))


class Normalizer:
    """A utility to normalize different types of scores."""
    
    @staticmethod
    def higher_is_better(scores: List[float]) -> List[float]:
        """Normalizes scores where a higher value is better (e.g., quality)."""
        min_val = min(scores)
        max_val = max(scores)
        if max_val == min_val:
            return [1.0] * len(scores)
        return [(s - min_val) / (max_val - min_val) for s in scores]

    @staticmethod
    def lower_is_better(scores: List[float]) -> List[float]:
        """Normalizes scores where a lower value is better (e.g., cost)."""
        # Invert the logic of higher_is_better
        normalized = Normalizer.higher_is_better(scores)
        return [1.0 - s for s in normalized]

class HybridScoringEngine:
    """
    Calculates a weighted score for each candidate model based on multiple signals.
    """
    def __init__(self, model_registry, weights: Dict[str, float]):
        """
        Initializes the engine with a model registry and scoring weights.

        Args:
            model_registry: An instance of ModelRegistry.
            weights (Dict[str, float]): A dictionary defining the importance
                                        of each scoring parameter.
        """
        self.registry = model_registry
        self.weights = weights
        self._validate_weights()
        
    def _validate_weights(self):
        """Ensures the provided weights sum to approximately 1.0."""
        total_weight = sum(self.weights.values())
        if not (0.99 < total_weight < 1.01):
            raise ValueError(f"Weights must sum to 1.0, but got {total_weight}")

    def score_models(self, semantic_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Scores all models based on the available signals.

        Args:
            semantic_result (Dict[str, Any]): The output from the SemanticRouter.

        Returns:
            A list of dictionaries, each containing a model and its detailed scores,
            sorted from best to worst.
        """
        models = self.registry.get_all_models()
        candidates = []

        # Collect raw scores for all models
        for model_id, specs in models.items():
            # Semantic Score: High if the model matches the semantic recommendation
            semantic_score = semantic_result['similarity_score'] if model_id == semantic_result['metadata']['ideal_model'] else (1 - semantic_result['similarity_score']) * 0.1

            # Cost Score: Lower is better. Rough estimate of total cost.
            cost_score = specs['cost_per_million_tokens']['input'] + specs['cost_per_million_tokens']['output']
            
            # Latency Score: Lower is better. Map tier to a number.
            latency_map = {"very_low": 1, "low": 2, "medium": 3, "high": 4}
            latency_score = latency_map.get(specs['latency_tier'], 99)

            candidates.append({
                "model_id": model_id,
                "scores_raw": {
                    "semantic": semantic_score,
                    "quality": specs['quality_score'],
                    "cost": cost_score,
                    "latency": latency_score
                }
            })
            
        # 2. Normalize the scores across all candidates
        raw_scores_by_param = {param: [c['scores_raw'][param] for c in candidates] for param in self.weights.keys()}
        
        normalized_scores = {
            "semantic": Normalizer.higher_is_better(raw_scores_by_param["semantic"]),
            "quality": Normalizer.higher_is_better(raw_scores_by_param["quality"]),
            "cost": Normalizer.lower_is_better(raw_scores_by_param["cost"]),
            "latency": Normalizer.lower_is_better(raw_scores_by_param["latency"])
        }

        # 3. Calculate final weighted score
        for i, candidate in enumerate(candidates):
            final_score = 0
            candidate['scores_normalized'] = {}
            for param, weight in self.weights.items():
                norm_score = normalized_scores[param][i]
                final_score += weight * norm_score
                candidate['scores_normalized'][param] = norm_score
            candidate['final_score'] = final_score

        # 4. Sort by the final score
        return sorted(candidates, key=lambda x: x['final_score'], reverse=True)
