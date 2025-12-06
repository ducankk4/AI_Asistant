from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingAPI(ABC):
    def __init__(self, api_key:str, model:str):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def embed_query(self, query : str) -> List[float]:
        pass

    @abstractmethod
    def embed_documents(self, texts : List[str]) -> List[List[float]]:
        pass
    