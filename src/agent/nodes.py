from src.agent.model import RAGState, QueryRouting, QueryRewrite, QueryAnalysis, FinalState
from src.prompts.rag_prompts import RESPONSE_PROMPT, REWRITE_PROMPT, ROUTING_PROMPT, QUERY_ANALYSIS_PROMPT, FINAL_RESPONSE_PROMPT
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing import List, Dict
from src.logger import logger
from src.config import REWRITE_MODEL, ROUTING_MODEL, RERANKING_MODEL, RESPONSE_MODEL, GROQ_API_KEY, ANALYSIS_MODEL, GOOGLE_API_KEY
# from src.main import LaptopRAG
from src.vector_store.chroma import ChromaVectorStore
from src.config import CollectionNames
import asyncio

class RAGNodes:
    def __init__(self):
        # self.laptop_rag = LaptopRAG()
        self.collection_names = CollectionNames()
        self.chroma_vector_store = ChromaVectorStore()

    def query_rewrite_node(self, state: RAGState) -> RAGState:
        """node for query rewrite"""
        query = state['query']
        chat_history = state.get('chat_history', "")
        need_rewrite = state.get('need_rewrite', False)

        # implement rewrite chain
        rewrite_llm = ChatGoogleGenerativeAI(model= REWRITE_MODEL,
                               max_tokens=None,
                                api_key= GOOGLE_API_KEY)
        rewrite_chain = ChatPromptTemplate.from_template(REWRITE_PROMPT) | rewrite_llm.with_structured_output(QueryRewrite)

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
        routing_llm = ChatGoogleGenerativeAI(model= ROUTING_MODEL,     
                                 api_key= GOOGLE_API_KEY)
        routing_chain = ChatPromptTemplate.from_template(ROUTING_PROMPT) | routing_llm.with_structured_output(QueryRouting)

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
        """node for laptop retrieve docs"""
        rewrite_query = state['rewrited_query']
        logger.info("Retrieving documents from laptop collection...")
        try:
            print(rewrite_query)
            laptop_collection = self.chroma_vector_store.load_collection(self.collection_names.LAPTOP_COLLECTION_NAME)
            results = self.chroma_vector_store.hybrid_search(rewrite_query, laptop_collection)
            print(results)
            # logger.info(f"retrieved {len(results)} documents from laptop collection")
            logger.info(f"retrieved {len(results)} documents from laptop collection")
            logger.info(f"type of results : {type(results)}")
            state.update({
                'retrived_docs': results
            })
        except Exception as e:
            logger.error(f"Error retrieving documents from laptop collection: {e}")
            state.update({
                'retrived_docs': []
            })
        return state
    
    def csbh_retrieve_node(self, state: RAGState) -> RAGState:
        """node for chinh sach bao hanh retrieve docs"""
        rewrite_query = state['rewrited_query']
        logger.info("Retrieving documents from laptop collection...")
        csbh_collection = self.chroma_vector_store.load_collection(self.collection_names.CSBH_COLLECTION_NAME)
        results = self.chroma_vector_store.hybrid_search(query= rewrite_query, vector_store= csbh_collection)
        logger.info(f"retrieved {len(results)} documents from csbh collection")
        logger.info(f"type of results : {type(results)}")

        state.update({
            'retrived_docs': results
        })
        return state
    
    def csdt_retrieve_node(self, state: RAGState) -> RAGState:
        """node for chinh sach doi tra retrieve docs"""
        rewrite_query = state['rewrited_query']
        logger.info("Retrieving documents from laptop collection...")
        csdt_collection = self.chroma_vector_store.load_collection(self.collection_names.CSDT_COLLECTION_NAME)
        results = self.chroma_vector_store.hybrid_search(query= rewrite_query, vector_store= csdt_collection)
        logger.info(f"retrieved {len(results)} documents from csdt collection")
        logger.info(f"type of results : {type(results)}")
        state.update({
            'retrived_docs': results
        })
        return state
    
    def csvc_retrieve_node(self, state: RAGState) -> RAGState:
        """node for chinh sach van chuyen retrieve docs"""
        rewrite_query = state['rewrited_query']
        logger.info("Retrieving documents from laptop collection...")
        csvc_collection = self.chroma_vector_store.load_collection(self.collection_names.CSVC_COLLECTION_NAME)
        results = self.chroma_vector_store.hybrid_search(query= rewrite_query, vector_store= csvc_collection)
        logger.info(f"retrieved {len(results)} documents from csvc collection")
        logger.info(f"type of results : {type(results)}")
        state.update({
            'retrived_docs': results
        })
        return state
    
    def generate_node(self, state: RAGState) -> RAGState:
        """node for response generation"""
        retrieved_docs = state.get('retrived_docs', [])
        rewrited_query = state['rewrited_query']
        query = state['query']
        if not retrieved_docs:
            logger.info("No documents retrieved, skipping response generation.")
            context = "không tìm thấy thông tin liên quan."
        else:
            context = "\n".join([doc.page_content for doc in retrieved_docs])

        # implement response generation chain
        response_llm = ChatGoogleGenerativeAI(model= RESPONSE_MODEL,
                                  api_key= GOOGLE_API_KEY)
        reponse_chain = ChatPromptTemplate.from_template(RESPONSE_PROMPT) | response_llm

        logger.info("Generating response...")
        response = reponse_chain.invoke({
            'query': query,
            'context': context
        })
        state.update({
            'subquery_answers': response.content
        })

        return state

class FinalNodes:
    def __init__(self):
        from src.agent.graphs import RAGGraph
        self.rag_processor = RAGGraph().implement_graph()

    def query_analysis_node(self, state: FinalState) -> FinalState:
        query_analysis_prompt = ChatPromptTemplate.from_template(QUERY_ANALYSIS_PROMPT)
        llm = ChatGoogleGenerativeAI(model= ANALYSIS_MODEL,
                       temperature=0,
                         api_key= GOOGLE_API_KEY)
        analysis_chain = query_analysis_prompt | llm.with_structured_output(QueryAnalysis)

        logger.info("Analyzing query...")

        results: QueryAnalysis = analysis_chain.invoke({
            'query': state['query']
        })

        print(f"response from query_analysis_node: {results}")

        return{
            **state,
            'original_query': state['query'],
            'need_decomposition': results.need_decomposition,
            'sub_queries' : results.sub_queries,
            'execution_plan': results.execution_plan,
        }
    
    async def process_queries(self, state: FinalState) -> FinalState:
        """processing query by using asyncio"""
        need_decomposition = state['need_decomposition']
        sub_queries = state.get('sub_queries', [state['query']])
        execution_plan = state['execution_plan']
        query = state['query']

        query_results = {}

        if not need_decomposition :
            print("processing single query  from rag_processor")
            rag_result = await self._process_single_query(query)
            query_results[rag_result['query']] = rag_result['retrived_docs']

            return {
                **state,
                'all_retrieved_docs' : rag_result['retrived_docs'],
                'query_results' : query_results
            }
        
        elif execution_plan == 'parallel':
            print(f"processing {len(sub_queries)} query parallel")

            tasks = [self._process_single_query(query, need_rewrite= False) for query in sub_queries ]

            results = await asyncio.gather(*tasks)
            for result in results:
                query_results[result['query']] = result['retrived_docs']
            
            return {**state,
                    'query_results' : query_results}
            
        else:
            print(f"processing {len(sub_queries)} query sequentially")
            results = await self._process_single_query(query=query, need_rewrite= True)
            query_results[query] = results['retrived_docs']

            return {
                **state,
                'query_results' : query_results
            }

    
    async def _process_single_query(self, query: str, need_rewrite: bool = False):
        rag_state: RAGState = {
            "query": query,
            "chat_history": [],
            "rewrited_query": "",
            "reason_for_rewrite": "",
            "reason_for_routing": "",
            "need_rewrite": need_rewrite,
            "routing_decision": "",
            "retrived_docs": [],
            "reranked_docs": [],
            "subquery_answers": ""
        }
        
        rag_result: RAGState = await self.rag_processor.ainvoke(rag_state)

        return rag_result
    
    def dict_to_text(self, query_results: Dict[str, List[Document]]) -> str:
        """convert dict of query results from rag processor to text format"""
        query_combined = ""
        for sub_query, docs in query_results.items():
            doc_combined = "\n".join([f"- {doc.page_content}" for doc in docs])
            query_combined += f"**Câu hỏi con: {sub_query}\n Thông tin liên quan:\n{doc_combined}\n----------------\n"

        return query_combined.strip()
        
    def final_generate_node(self, state: FinalState) -> FinalState:
        """final node for answer user question"""
        original_quey = state['original_query']
        need_decompositon = state['need_decomposition']
        query_results = state['query_results']
        all_docs = state.get('all_retrieved_docs', [])

        # if all_docs:
        response_llm = ChatGoogleGenerativeAI(model= RESPONSE_MODEL, 
                                api_key= GOOGLE_API_KEY)
        reponse_chain = ChatPromptTemplate.from_template(FINAL_RESPONSE_PROMPT) | response_llm
        query_combined = self.dict_to_text(query_results)
        logger.info("Generating final response...")
        final_response = reponse_chain.invoke({
            'original_query': original_quey,
            'query_combined': query_combined
        })

        return {
            **state,
            'final_answer': final_response.content
        }
        
    

    


# if __name__ == "__main__":
#     rag_nodes = RAGNodes()
#     rag_state: RAGState = {
#         "query": "ANh tên là Đức đẹp trai năm nay 24 tuổi nhà ở Hà nội anh 24 tuổi nên muốn tìm 1 cái laptop chơi game khỏe giá trung bình",
#         "chat_history": [],
#         "rewrited_query": "",
#         "reason_for_rewrite": "",
#         "reason_for_routing": "",
#         "need_rewrite": True,
#         "routing_decision": "",
#         "retrived_docs": [],
#         "reranked_docs": [],
#         "subquery_answers": ""
#     }
#     result = rag_nodes.query_rewrite_node(rag_state)
#     print(result)

    
    



        


