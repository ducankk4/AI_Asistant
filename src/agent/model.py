from pydantic import BaseModel, Field
from typing import List, TypedDict, Literal
from langchain_core.documents import Document

class RAGState(TypedDict):
    """state of RAG agent"""
    query: str
    rewrited_query: str
    reason_for_rewrite: str
    reason_for_routing: str
    need_rewrite: bool
    num_retries: int
    routing_decision: str
    retrived_docs: List[Document]
    reranked_docs: List[Document]
    chat_history: List[str]
    subquery_answers: str

class QueryRouting(BaseModel):
    """query routing for single query"""
    collection_needed: Literal["laptop", "csbh", "csdt", "csvc"] = Field(
        description= "collections that should be searched"
    )
    reasoning: str = Field(
        description= "reasoning for choosing the collection"
    )

class QueryRewrite(BaseModel):
    """Rewrited query for better retrieval"""
    rewrited_query: str = Field(
        description= "the improved version of the original query for better retrieval"
    )
    reasoning: str = Field(
        description= "reasoning for the improved"
    )

