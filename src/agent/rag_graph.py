from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from config import ROUTING_MODEL, RERANKING_MODEL, RESPONSE_MODEL
from logger import logger

class RAGGraph:
    def __init__(self):
        self.routing_model = ROUTING_MODEL
        self.reranking_model = RERANKING_MODEL
        self.response_model = RESPONSE_MODEL

    