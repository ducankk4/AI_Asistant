from model import RAGState, QueryRouting, QueryRewrite
from prompts.rag_prompts import RESPONSE_PROMPT, REWRITE_PROMPT, ROUTING_PROMPT
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from logger import logger
from config import REWRITE_MODEL, ROUTING_MODEL, RERANKING_MODEL, RESPONSE_MODEL, GROQ_API_KEY
from main import LaptopRAG

class RAGNodes:
    def __init__(self):
        self.laptop_rag = LaptopRAG()

    def query_rewrite_node(self, state: RAGState) -> RAGState:
        """node for query rewrite"""
        query = state['query']
        chat_history = state['chat_history']
        need_rewrite = state.get('need_rewrite', False)
        reason_for_rewrite = state.get('reason_for_rewrite', '')

        # implement rewrite chain
        rewrite_llm = ChatGroq(model= REWRITE_MODEL, api_key= GROQ_API_KEY)
        rewrite_chain = ChatPromptTemplate(REWRITE_PROMPT) | rewrite_llm.with_structured_output(QueryRewrite)

        if not need_rewrite:
            state['rewrited_query'] = query
            return state
        
        logger.info("Rewriting query...")

        rewrite_result = rewrite_chain.invoke({
            'query': query,
            'chat_history': chat_history
        })
        logger.info(f"original query: {query}")
        logger.info(f"rewrited query: {rewrite_result.rewrited_query}")
        logger.info(f"reasoning for rewrite: {rewrite_result.reasoning}")

        state.update({
            'rewrited_query': rewrite_result.rewrited_query,
            'reason_for_rewrite': rewrite_result.reasoning,
            'need_rewrite': False
        })
        return state
    
    def query_routing_node(self, state: RAGState) -> RAGState:
        """node for query routing"""
        rewrited_query = state['rewrited_query']
        
        # implement routing chain
        routing_llm = ChatGroq(model= ROUTING_MODEL, api_key= GROQ_API_KEY)
        routing_chain = ChatPromptTemplate(ROUTING_PROMPT) | routing_llm.with_structured_output(QueryRouting)

        logger.info("Routing query...")
        routing_result = routing_chain.invoke({
            'query': rewrited_query
        })
        
        logger.info(f"collection needed: {routing_result.collection_needed}")
        logger.info(f"reasoning for routing: {routing_result.reasoning}")

        state.update({
            'routing_decision': routing_result.collection_needed,
            'reason_for_routing': routing_result.reasoning
        })
        return state
    
    def laptop_retrieve_node(self, state: RAGState) -> RAGState:
        """node for laptop response generation"""
        rewrite_query = state['rewrited_query']

        results = self.laptop_rag.similarity_response(rewrite_query)
        state.update({
            'retrived_docs': results
        })
        return state
    
    def generation_node(self, state: RAGState) -> RAGState:
        """node for response generation"""
        retrieved_docs = state['retrived_docs']
        rewrited_query = state['rewrited_query']
        query = state['query']
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # implement response generation chain
        response_llm = ChatGroq(model= RESPONSE_MODEL, api_key= GROQ_API_KEY)
        reponse_chain = ChatPromptTemplate(RESPONSE_PROMPT) | response_llm

        logger.info("Generating response...")
        response = reponse_chain.invoke({
            'query': query,
            'context': context
        })
        state.update({
            'subquery_answers': response.content
        })

        return state

        


