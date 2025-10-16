import streamlit as st
import json
import os
from typing import TypedDict, List, Dict, Any
import pandas as pd
import sys
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

# Page Configuration
st.set_page_config(
    page_title="Intelligent LLM Router",
    page_icon="🧠",
    layout="wide"
)

#  Graph State 
class GraphState(TypedDict):
    prompt: str
    classifier_result: ClassificationResult
    semantic_match: dict
    ranked_models: List[Dict[str, Any]]
    final_model_id: str
    final_response: str


@st.cache_resource
def initialize_services():
    print("Initializing services...")

    if "GOOGLE_API_KEY" not in os.environ:
        st.error("ERROR: Please set the GOOGLE_API_KEY environment variable before starting the app.")
        st.stop()
    
    classifier = NvidiaClassifier()
    semantic_router = SemanticRouter()
    model_registry = ModelRegistry()
    return classifier, semantic_router, model_registry

classifier, semantic_router, model_registry = initialize_services()

# --- 3. LangGraph Builiding ---

def build_langgraph_app():
    """Builds and compiles the LangGraph application."""
    
    SCORING_WEIGHTS = {
        "semantic": 0.5, "quality": 0.2, "cost": 0.15, "latency": 0.15,
    }
    scoring_engine = HybridScoringEngine(model_registry, weights=SCORING_WEIGHTS)

    # --- Node Definitions ---
    def classify_node(state: GraphState) -> Dict[str, Any]:
        result = classifier.classify(state['prompt'])
        return {"classifier_result": result}

    def semantic_search_node(state: GraphState) -> Dict[str, Any]:
        match = semantic_router.find_best_match(state['prompt'], state['classifier_result'].task_type_1)
        if match is None:
            raise ValueError(f"Semantic search failed: No examples found for task type '{state['classifier_result'].task_type_1}'")
        return {"semantic_match": match}

    def scoring_node(state: GraphState) -> Dict[str, Any]:
        ranked_models = scoring_engine.score_models(state['semantic_match'])
        best_model_id = ranked_models[0]['model_id']
        return {"ranked_models": ranked_models, "final_model_id": best_model_id}

    def api_call_node(state: GraphState) -> Dict[str, Any]:
        try:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            model = genai.GenerativeModel(state['final_model_id'])
            response = model.generate_content(state['prompt'])
            if response.parts:
                return {"final_response": response.text}
            else:
                return {"final_response": "The model did not provide a response, possibly due to safety filters."}
        except Exception as e:
            return {"final_response": f"Error calling model {state['final_model_id']}: {e}"}

    # --- Graph Assembly ---
    workflow = StateGraph(GraphState)
    workflow.add_node("classifier", classify_node)
    workflow.add_node("semantic_search", semantic_search_node)
    workflow.add_node("scorer", scoring_node)
    workflow.add_node("api_caller", api_call_node)
    workflow.set_entry_point("classifier")
    workflow.add_edge("classifier", "semantic_search")
    workflow.add_edge("semantic_search", "scorer")
    workflow.add_edge("scorer", "api_caller")
    workflow.add_edge("api_caller", END)
    
    return workflow.compile()

app = build_langgraph_app()


# --- 4. Streamlit UI Components ---

# Initialize session state for chat history and routing details
if "messages" not in st.session_state:
    st.session_state.messages = []
if "routing_details" not in st.session_state:
    st.session_state.routing_details = None
if "analytics" not in st.session_state:
    st.session_state.analytics = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost": 0.0,
        "comparative_costs": {model_id: 0.0 for model_id in model_registry.get_all_models().keys()}
    }


st.title("🧠 Intelligent LLM Router")
st.caption("An application that intelligently routes your prompt to the most suitable Gemini model.")

# --- Sidebar for Routing Details ---
with st.sidebar:
    st.header("🔍 Routing Dashboard")
    if st.session_state.routing_details:
        details = st.session_state.routing_details
        
        st.subheader("1. Prompt Classification")
        st.metric("Identified Task", details['classifier_result'].task_type_1)
        
        st.subheader("2. Semantic Search")
        st.metric("Best Match Recommendation", details['semantic_match']['metadata']['ideal_model'])
        st.metric("Similarity Score", f"{details['semantic_match']['similarity_score']:.2f}")

        st.subheader("3. Hybrid Scoring & Ranking")
        st.dataframe(details['ranked_models'],
                     column_config={
                         "model_id": "Model",
                         "final_score": st.column_config.ProgressColumn("Final Score", format="%.3f", min_value=0, max_value=1),
                         "semantic_norm": "Semantic",
                         "quality_norm": "Quality",
                         "cost_norm": "Cost",
                         "latency_norm": "Latency",
                     },
                     hide_index=True,
                     use_container_width=True)

        st.subheader("🏆 Final Decision")
        st.success(f"**Selected Model:** {details['final_model_id']}")

    else:
        st.info("Send a message to see the live routing details here.")

    # st.header("📊 Session Analytics")
    # if st.session_state.analytics['total_input_tokens'] > 0:
    #     analytics = st.session_state.analytics
    #     col1, col2 = st.columns(2)
    #     col1.metric("Input Tokens", f"{analytics['total_input_tokens']:,}")
    #     col2.metric("Output Tokens", f"{analytics['total_output_tokens']:,}")
        
    #     st.metric("Router Cost (Actual)", f"${analytics['total_cost']:.6f}")

    #     st.subheader("Cost Comparison")
        
    #     comp_data = {
    #         "Strategy": ["Router (Actual Cost)", "Always Pro", "Always Flash", "Always Flash-Lite"],
    #         "Total Cost ($)": [
    #             analytics['total_cost'],
    #             analytics['comparative_costs']['gemini-2.5-pro'],
    #             analytics['comparative_costs']['gemini-2.5-flash'],
    #             analytics['comparative_costs']['gemini-2.5-flash-lite'],
    #         ]
    #     }
    #     df = pd.DataFrame(comp_data)
    #     st.dataframe(df, hide_index=True, width='stretch')
        
    #     savings = analytics['comparative_costs']['gemini-2.5-pro'] - analytics['total_cost']
    #     if savings > 0:
    #         st.success(f"**Savings vs 'Always Pro':** ${savings:.6f}")
    #     else:
    #         st.warning(f"**Cost vs 'Always Pro':** ${abs(savings):.6f} more expensive")
    # else:
    #     st.info("Analytics will appear here after your first message.")


# --- Main Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Analyzing prompt and routing to the best model..."):
        inputs = {"prompt": prompt}
        final_state = app.invoke(inputs)
        
        st.session_state.routing_details = final_state
        
        response = final_state.get("final_response", "Sorry, something went wrong.")
        
        # --- Analytics Calculation ---
        # if response and isinstance(response, str) and not response.startswith("The model did not"):
        #     try:
        #         model_client = genai.GenerativeModel(final_state['final_model_id'])
        #         input_tokens = model_client.count_tokens(prompt).total_tokens
        #         output_tokens = model_client.count_tokens(response).total_tokens

        #         for model_id, model_data in model_registry.get_all_models().items():
        #             input_cost = (input_tokens / 1_000_000) * model_data['pricing']['input_cost']['text']
        #             output_cost = (output_tokens / 1_000_000) * model_data['pricing']['output_cost']['text']
        #             turn_cost = input_cost + output_cost
        #             st.session_state.analytics['comparative_costs'][model_id] += turn_cost
        #             if model_id == final_state['final_model_id']:
        #                 st.session_state.analytics['total_cost'] += turn_cost

        #         st.session_state.analytics['total_input_tokens'] += input_tokens
        #         st.session_state.analytics['total_output_tokens'] += output_tokens
        #     except Exception as e:
        #         print(f"ANALYTICS FAILED: {e}") 
        #         st.warning("Could not calculate analytics for this turn.")

        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display assistant response and refresh the sidebar
        with st.chat_message("assistant"):
            st.markdown(response)
        st.rerun()
