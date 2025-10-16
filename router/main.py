"""
The main entry point and orchestrator for the LLM Routing pipeline.
This script uses LangGraph to define a stateful graph that executes each
component of the routing logic in a clear, sequential manner.
"""
import json
from typing import TypedDict, List, Dict, Any
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from langgraph.graph import StateGraph, END
import google.generativeai as genai

from classifier.classifier_service import NvidiaClassifier, ClassificationResult
from semantic.semantic_search import SemanticRouter
from model.model_registry import ModelRegistry
from engine.scoring_engine import HybridScoringEngine

# --- 1. Define the State for our Graph ---

class GraphState(TypedDict):
    """
    Represents the state of our routing graph.

    Attributes:
        prompt (str): The initial user prompt.
        classifier_result (ClassificationResult): The output from the NVIDIA classifier.
        semantic_match (dict): The best match found by the semantic router.
        ranked_models (list): The final list of models, sorted by score.
        final_model_id (str): The ID of the chosen model for the API call.
        final_response (str): The simulated response from the chosen LLM.
    """
    prompt: str
    classifier_result: ClassificationResult
    semantic_match: dict
    ranked_models: List[Dict[str, Any]]
    final_model_id: str
    final_response: str


# --- 2. Initialize our Services ---

classifier = NvidiaClassifier()
semantic_router = SemanticRouter()
model_registry = ModelRegistry()

# Define the weights for the scoring engine it can be adjusted as needed and can be dynamic too based on the use case
SCORING_WEIGHTS = {
    "semantic": 0.5,
    "quality": 0.2,
    "cost": 0.15,
    "latency": 0.15,
}
scoring_engine = HybridScoringEngine(model_registry, weights=SCORING_WEIGHTS)


# --- 3. Define the Nodes of our Graph ---
# Each node is a function that takes the current state, performs an action,
# and returns a dictionary with the updated state values.

def classify_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 1: Classifies the user's prompt to understand its intent.
    """
    print("--- Step 1: Classifying Prompt ---")
    prompt = state['prompt']
    result = classifier.classify(prompt)
    print(f"Task Type: {result.task_type_1}")
    return {"classifier_result": result}

def semantic_search_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 2: Performs a filtered semantic search based on the classification.
    """
    print("\n--- Step 2: Performing Filtered Semantic Search ---")
    prompt = state['prompt']
    task_type = state['classifier_result'].task_type_1
    match = semantic_router.find_best_match(prompt, task_type)
    
    if match is None:
        raise ValueError(f"Semantic search failed: No examples found for task type '{task_type}'")
    
    print(f"Found match: Recommends '{match['metadata']['ideal_model']}' with score {match['similarity_score']:.2f}")
    return {"semantic_match": match}

def scoring_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 3: Scores all candidate models and selects the best one.
    """
    print("\n--- Step 3: Running Hybrid Scoring Engine ---")
    semantic_match = state['semantic_match']
    ranked_models = scoring_engine.score_models(semantic_match)
    best_model_id = ranked_models[0]['model_id']
    
    print(f"Best model selected: {best_model_id}")
    return {"ranked_models": ranked_models, "final_model_id": best_model_id}

def api_call_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 4: Makes the final API call to the selected model.
    (This is a simulation for demonstration purposes).
    """
    print("\n--- Step 4: Calling Selected LLM API (Simulation) ---")
    prompt = state['prompt']
    model_id = state['final_model_id']
    

    print(f"Calling model '{model_id}' with prompt: '{prompt}'")

    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

        print(f"Calling model '{model_id}' with prompt: '{prompt}'")
        
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(prompt)

        print("API call successful.")
        return {"final_response": response.text}
    except Exception as e:
        print(f"An error occurred during the API call: {e}")
        return {"final_response": f"Error calling model {model_id}: {e}"}
    


# --- 4. Assemble the Graph ---
# We define the structure and flow of our application.

workflow = StateGraph(GraphState)

# Add the nodes to the graph
workflow.add_node("classifier", classify_node)
workflow.add_node("semantic_search", semantic_search_node)
workflow.add_node("scorer", scoring_node)
workflow.add_node("api_caller", api_call_node)

# Define the edges
workflow.set_entry_point("classifier")
workflow.add_edge("classifier", "semantic_search")
workflow.add_edge("semantic_search", "scorer")
workflow.add_edge("scorer", "api_caller")
workflow.add_edge("api_caller", END)

# Compile the graph
app = workflow.compile()


# --- 5. Run the Full Pipeline ---
if __name__ == "__main__":

    if "GOOGLE_API_KEY" not in os.environ:
        print(" ERROR: Please set the GOOGLE_API_KEY environment variable.")
        print("You can get a key from Google AI Studio and run 'export GOOGLE_API_KEY=\"YOUR_KEY\"'")
        exit(1)

    user_prompt = "classify the following text sentiment as positive, negative, or neutral: 'I love the new features in the latest update!'"
    
    print("🚀 Starting LLM Router Pipeline...")
    

    inputs = {"prompt": user_prompt}
    
    final_state = app.invoke(inputs)


    print("\n✅ Pipeline Finished Successfully!")
    print("\n--- Final Result ---")
    print(f"Initial Prompt: {final_state['prompt']}")
    print(f"Chosen Model: {final_state['final_model_id']}")
    print(f"Final Response: {final_state['final_response']}")
    
    print("\n--- Detailed Scoring Breakdown ---")
    print(json.dumps(final_state['ranked_models'], indent=2))
