from src.embedding.base import BaseEmbeddingAPI
from typing import List
from google import genai
# from config import GOOGLE_API_KEY, GOOGLE_EMBEDDING_MODEL

class GoogleEmbedding(BaseEmbeddingAPI):
    
    def __init__(self, api_key, model):
        super().__init__(api_key,model)
    
    def initialize_client(self):
        client = genai.Client(api_key=self.api_key)
        return client

    def embed_query(self, query: str) -> List[float]:
        client = self.initialize_client()
        response = client.models.embed_content(
            model = self.model,
            contents= query
        )
        return response.embeddings[0].values
    
    def embed_documents(self, texts : List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embed = self.embed_query(text)
            embeddings.append(embed)
        return embeddings
    

    