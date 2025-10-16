"""
Provides a semantic search capability to find similar prompts from a
curated database. This is the second major component of our routing pipeline.
"""
import json
import chromadb
from sentence_transformers import SentenceTransformer

class SemanticRouter:
    """
    Manages loading a prompt database, creating embeddings, and performing
    filtered similarity searches using a vector database.
    """
    COLLECTION_NAME = "prompt_examples"

    def __init__(self, db_path: str = "semantic/prompt_database.json"):
        """
        Initializes the router, loads the database, and sets up the
        embedding model and vector database client.

        Args:
            db_path (str): Path to the JSON file containing prompt examples.
        """
        print("Initializing Semantic Router...")
        self.db_data = self._load_database(db_path)
        
        # Using an open-source model for creating embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.Client()
        self._setup_collection()
        print("Semantic Router initialized and database loaded.")

    def _load_database(self, db_path: str):
        """Loads the prompt examples from a JSON file."""
        try:
            with open(db_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Database file not found at {db_path}")
            return []

    def _setup_collection(self):
        """
        Creates or gets a ChromaDB collection and populates it with our
        prompt examples if it's empty.
        """
        collection = self.chroma_client.get_or_create_collection(name=self.COLLECTION_NAME)
        
        if collection.count() == 0 and self.db_data:
            print(f"Populating ChromaDB collection '{self.COLLECTION_NAME}'...")
            
            prompt_texts = [item['prompt_text'] for item in self.db_data]
            metadatas = [{'task_type': item['task_type'], 'ideal_model': item['ideal_model']} for item in self.db_data]
            ids = [item['prompt_id'] for item in self.db_data]

            # Generate embeddings for all prompts in the database
            embeddings = self.embedding_model.encode(prompt_texts).tolist()

            collection.add(
                embeddings=embeddings,
                documents=prompt_texts,
                metadatas=metadatas,
                ids=ids
            )
            print("Collection populated successfully.")
        
        self.collection = collection

    def find_best_match(self, user_prompt: str, task_type: str, n_results: int = 2):
        """
        Finds the most similar prompt(s) in the database, filtered by task type.

        Args:
            user_prompt (str): The incoming user prompt.
            task_type (str): The task type determined by the classifier.
            n_results (int): The number of similar prompts to return.

        Returns:
            A dictionary containing the best match's metadata and similarity score,
            or None if no match is found.
        """
        if not self.collection:
            return None

        # generate embedding for the user's prompt
        query_embedding = self.embedding_model.encode([user_prompt]).tolist()

        # Query ChromaDB with a metadata filter
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where={"task_type": task_type} # this will filter results by task_type that we get from classifier
        )
        
        # Check if any results were returned
        if not results or not results['ids'][0]:
            return None

        # Extract the top result's info
        best_match_id = results['ids'][0][0]
        best_match_metadata = results['metadatas'][0][0]
        best_match_similarity = 1 - results['distances'][0][0]

        return {
            "id": best_match_id,
            "metadata": best_match_metadata,
            "similarity_score": best_match_similarity
        }

