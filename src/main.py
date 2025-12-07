from vector_store.chroma import ChromaVectorStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
from config import FINAL_CSBH_DATA, FINAL_CSDT_DATA,FINAL_CSVC_DATA,FINAL_LAPTOP_DATA, LAPTOP_COLLECTION_NAME, CSBH_COLLECTION_NAME, CSDT_COLLECTION_NAME, CSVC_COLLECTION_NAME
from logger import logger

class LaptopRAG:
    def __init__(self):
        self.chroma_vector_store = ChromaVectorStore()
        self.vector_store = None

    def get_collection(self):
        vector_store = self.chroma_vector_store.get_or_create_collection(
            collection_name= LAPTOP_COLLECTION_NAME,
            data_path= FINAL_LAPTOP_DATA
        )
        return vector_store
    
    def ensure_vector_store(self):
        if self.vector_store is None:
            self.vector_store = self.get_collection()
    
    def similarity_retrieve(self, query: str):
        self.ensure_vector_store()
        chroma_vector_store = self.chroma_vector_store  
        results = chroma_vector_store.similar_search(query, self.vector_store)

        return results
    
    def hybrid_retrieve(self, query: str):
        self.ensure_vector_store()
        chroma_vector_store = self.chroma_vector_store
        results = chroma_vector_store.hybrid_search(query, self.vector_store)

        return results
    
    def doc_to_text(self, docs: List[Document]) -> str:
        pass

    def similar_response(self, query: str):
        results = self.similarity_retrieve(query)
        texts = self.doc_to_text(results)

        return texts



    





# if __name__ == "__main__":
#     query = "CPU của laptop Dell Inspiron 14 5441 là gì?"
#     results = retrieve(query)
#     print(results)




    # with open(FINAL_LAPTOP_DATA, "r", encoding="utf-8") as f:
    #     laptop_text = f.read()
    # laptop_collection = chroma_vector_store.initialize_collection(
    #     collection_name= LAPTOP_COLLECTION_NAME,
    #     text = laptop_text
    # )
    # with open(FINAL_CSBH_DATA, "r", encoding="utf-8") as f:
    #     csbh_text = f.read()
    # csbh_collection = chroma_vector_store.initialize_collection(
    #     collection_name= CSBH_COLLECTION_NAME,
    #     text = csbh_text
    # )
    # with open(FINAL_CSDT_DATA, "r", encoding="utf-8") as f:
    #     csdt_text = f.read()
    # csdt_collection = chroma_vector_store.initialize_collection(
    #     collection_name= CSDT_COLLECTION_NAME,
    #     text = csdt_text
    # )
    # with open(FINAL_CSVC_DATA, "r", encoding="utf-8") as f:
    #     csvc_text = f.read()
    # csvc_collection = chroma_vector_store.initialize_collection(
    #     collection_name= CSVC_COLLECTION_NAME,  
    #     text = csvc_text
    # )

    