from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from src.agent.nodes import RAGNodes, FinalNodes
from src.agent.model import RAGState, FinalState
from typing import List, Literal
from src.config import ROUTING_MODEL, RERANKING_MODEL, RESPONSE_MODEL
from src.logger import logger

class RAGGraph:
    def __init__(self, memory_saver = None):
        self.graph_node = RAGNodes()
        self.memory_saver = memory_saver if memory_saver else MemorySaver()

    def route_to_collection(self, state: RAGState) -> Literal["laptop_retrieve_node", "csbh_retrieve_node", "csdt_retrieve_node", "csvc_retrieve_node"]:
        "route to the collection based on query routing node"
        collection_needed = state["routing_decision"]
        return f"{collection_needed}_retrieve_node"
    
    def implement_graph(self):
        # build the rag graph
        rag_graph = StateGraph(state_schema=RAGState)

        # add nodes
        rag_graph.add_node("query_rewrite_node", self.graph_node.query_rewrite_node)
        rag_graph.add_node("query_routing_node", self.graph_node.query_routing_node)
        rag_graph.add_node("laptop_retrieve_node", self.graph_node.laptop_retrieve_node)
        rag_graph.add_node("csbh_retrieve_node", self.graph_node.csbh_retrieve_node)
        rag_graph.add_node("csdt_retrieve_node", self.graph_node.csdt_retrieve_node)
        rag_graph.add_node("csvc_retrieve_node", self.graph_node.csvc_retrieve_node)
        rag_graph.add_node("generate_node", self.graph_node.generate_node)

        # set entry point which node to start in graph
        rag_graph.set_entry_point("query_rewrite_node")

        # add edges
        rag_graph.add_edge("query_rewrite_node", "query_routing_node")

        # add conditonal edges from routing node
        rag_graph.add_conditional_edges(
            "query_routing_node",
            self.route_to_collection,
            {
                "laptop_retrieve_node": "laptop_retrieve_node",
                "csbh_retrieve_node": "csbh_retrieve_node",
                "csdt_retrieve_node": "csdt_retrieve_node",
                "csvc_retrieve_node": "csvc_retrieve_node"
            }
        )

        # add edges to generation node
        rag_graph.add_edge("laptop_retrieve_node", "generate_node")
        rag_graph.add_edge("csbh_retrieve_node", "generate_node")
        rag_graph.add_edge("csdt_retrieve_node", "generate_node")
        rag_graph.add_edge("csvc_retrieve_node", "generate_node")

        rag_graph.add_edge("generate_node", END)

        # compile graph
        compiled_rag_graph = rag_graph.compile(checkpointer= self.memory_saver)

        return compiled_rag_graph

class FinalGraph:
    def __init__(self, memory_saver = None):
        # self.rag_processor = RAGGraph().implement_graph()
        self.final_node = FinalNodes()
        self.memory_saver = memory_saver if memory_saver else MemorySaver()
        # initialize stateGraph
        final_graph = StateGraph(state_schema= FinalState)

        # add nodes
        final_graph.add_node("query_analysis_node", self.final_node.query_analysis_node)
        final_graph.add_node("process_queries", self.final_node.process_queries)
        final_graph.add_node("final_generate_node", self.final_node.final_generate_node)

        # set entry point 
        final_graph.set_entry_point("query_analysis_node")

        # analyze -> process -> final generate
        final_graph.add_edge("query_analysis_node", "process_queries")
        final_graph.add_edge("process_queries", "final_generate_node")
        final_graph.add_edge("final_generate_node", END)

        # compile gragh
        compiled_final_graph = final_graph.compile(checkpointer= self.memory_saver)
        return compiled_final_graph
    

        

    

    