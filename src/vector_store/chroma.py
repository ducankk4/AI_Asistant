from langchain_core.documents import Document
from langchain_chroma.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from typing import List
from src.embedding.google_embed import GoogleEmbedding
from src.logger import logger
from src.config import rag_config, GOOGLE_API_KEY, GOOGLE_EMBEDDING_MODEL
import chromadb
import os

class ChromaVectorStore:
    def __init__(self):
        self.embedding_function = GoogleEmbedding(GOOGLE_API_KEY, GOOGLE_EMBEDDING_MODEL)
        self.rag_config = rag_config
        self.client = chromadb.PersistentClient(
            path= self.rag_config.persist_directory
        )
        logger.info(f"Initialized ChromaDB persistent client at: {self.rag_config.persist_directory}")

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
    
    def check_collection_exists(self, collection_name: str) -> bool:
        existing_collections = self.client.list_collections()
        for coll in existing_collections:
            if coll.name == collection_name:
                return True
        return False
    
    def delete_collection(self, collection_name: str):
        if self.check_collection_exists(collection_name):
            self.client.delete_collection(name= collection_name)
            logger.info(f"deleted collection: {collection_name}")
        else:
            logger.info(f"Collection {collection_name} does not exist, cant delete")
    
    def load_collection(self, collection_name: str) -> Chroma:
        vector_store = Chroma(
            client= self.client,
            embedding_function = self.embedding_function,
            collection_name= collection_name
        )
        logger.info(f"Loaded existing collection: {collection_name}")
        return vector_store
    
    def create_collection(self, collection_name: str, text: str) -> Chroma:
        documents = self.chunking(text)
        logger.info(f"_______________Total chunks created_____________: {len(documents)}")

        vector_store = Chroma.from_documents(
            documents=documents,
            client= self.client,
            embedding= self.embedding_function,
            collection_name= collection_name
        )
        logger.info(f"Created new collection: {collection_name}")

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


      