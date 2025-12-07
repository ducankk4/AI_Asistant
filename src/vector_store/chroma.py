from langchain_core.documents import Document
from langchain_chroma.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from typing import List
from embedding.google_embed import GoogleEmbedding
from logger import logger
from config import rag_config, GOOGLE_API_KEY, GOOGLE_EMBEDDING_MODEL
import chromadb
import os

class ChromaVectorStore:
    def __init__(self):
        self.embedding_function = GoogleEmbedding(GOOGLE_API_KEY, GOOGLE_EMBEDDING_MODEL)
        self.rag_config = rag_config

    def chunking(self, text: str) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.rag_config.chunk_size,
            chunk_overlap = self.rag_config.chunk_overlap,
        )
        docs = text_splitter.split_text(text)
        documents = self.text_to_documents(docs)
        return documents
    
    def text_to_documents(self, text: List[str]) -> List[Document]:
        documents = []
        for idx, text_chunk in enumerate(text):
            doc = Document(page_content=text_chunk, metadata= {"chunk_id": idx})
            documents.append(doc)
        return documents
    
    def load_collection(self, collection_name: str) -> Chroma:
        vector_store = Chroma(
            persist_directory= self.rag_config.persist_directory,
            embedding_function = self.embedding_function,
            collection_name= collection_name
        )
        logger.info(f"Loaded existing collection: {collection_name}")
        return vector_store
    
    def initialize_collection(self, collection_name: str, text: str) -> Chroma:
        documents = self.chunking(text)
        logger.info(f"_______________Total chunks created_____________: {len(documents)}")

        vector_store = Chroma.from_documents(
            documents=documents,
            persist_directory= self.rag_config.persist_directory,
            embedding= self.embedding_function,
            collection_name= collection_name
        )
        logger.info(f"Created new collection: {collection_name}")

        return vector_store
    
    def get_or_create_collection(self, collection_name: str, data_path: str) -> Chroma:
        try:
            vector_store = self.load_collection(collection_name)
    
        except Exception as e:
            logger.info(f"Collection not found, initializing new collections, error: {e}")
            with open(data_path, "r", encoding="utf-8") as f:
                laptop_text = f.read()
            vector_store = self.initialize_collection(
                collection_name= collection_name,
                text = laptop_text
            )
        return vector_store
    
    def similar_search(self, query: str, vector_store: Chroma) -> List[Document]:
        docs = vector_store.similarity_search(query= query, k= self.rag_config.top_k_result)
        return docs

    def hybrid_search(self, query: str, vector_store: Chroma) -> List[Document] :

        data = vector_store.get()
        documents =[
            Document(page_content= doc, metadata= metadata) 
            for doc, metadata in zip(data['documents'], data['metadatas'])
        ]
        bm25_retriever = BM25Retriever.from_documents(documents, search_kwargs = {"k": self.rag_config.top_k_result})
        mmr_retriever = vector_store.as_retriever(search_type = "mmr", search_kwargs = {"k": self.rag_config.top_k_result})
        similarity_retriever = vector_store.as_retriever(search_type= "similarity", search_kwargs = {"k": self.rag_config.top_k_result})

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, mmr_retriever, similarity_retriever],
            weights=[0.2,0.5,0.3]
        )
        results = ensemble_retriever.invoke(query)
        return results


      