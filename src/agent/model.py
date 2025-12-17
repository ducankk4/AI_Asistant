from pydantic import BaseModel, Field
from typing import List, TypedDict, Literal, Dict
from langchain_core.documents import Document

class RAGState(TypedDict):
    """state of RAG agent"""
    query: str
    rewrited_query: str
    reason_for_rewrite: str
    reason_for_routing: str
    need_rewrite: bool
    routing_decision: str
    retrived_docs: List[Document]
    reranked_docs: List[Document]
    chat_history: List[str]
    subquery_answers: str

class FinalState(TypedDict):
    """final state of AI Assistant"""
    query: str 
    original_query: str
    need_decomposition: bool
    execution_plan: str
    sub_queries: List[str] 
    query_results: Dict[str, List[Document]]
    all_retrieved_docs: List[Document]
    final_answer: str
    

class QueryRouting(BaseModel):
    """output schema for query routing node"""
    collection_needed: Literal["laptop", "csbh", "csdt", "csvc"] = Field(
        description= "collections that should be searched"
    )
    reasoning: str = Field(
        description= "reasoning for choosing the collection"
    )

class QueryRewrite(BaseModel):
    """output schema for query rewrite node"""
    rewrited_query: str = Field(
        description= "the improved version of the original query for better retrieval"
    )
    reasoning: str = Field(
        description= "reasoning for the improved"
    )

class QueryAnalysis(BaseModel):
    """output schema for query analysis node"""
    need_decomposition: bool = Field(
        description= "Câu hỏi có phức tạp và cần phải phân tách ra thành các câu hỏi nhỏ không"
    )
    sub_queries: List[str] = Field(
        description= "Danh sách các câu hỏi nhỏ đã được phân tách từ câu hỏi ban đầu",
        default_factory= list
    )
    execution_plan: Literal["parallel", "sequential"] = Field(
        description= "những câu hỏi đã được chia nhỏ nên được trả lời song song hay tuần tự",
        default= "parallel"
    )
    reasoning: str = Field(
        description= "Lý do cho việc phân tách câu hỏi"
    )



