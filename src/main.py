from vector_store.chroma import ChromaVectorStore
from config import FINAL_CSBH_DATA, FINAL_CSDT_DATA,FINAL_CSVC_DATA,FINAL_LAPTOP_DATA, LAPTOP_COLLECTION_NAME, CSBH_COLLECTION_NAME, CSDT_COLLECTION_NAME, CSVC_COLLECTION_NAME
from logger import logger

def retrieve(query: str):
    chroma_vector_store = ChromaVectorStore()
    collection_names = [LAPTOP_COLLECTION_NAME, CSBH_COLLECTION_NAME, CSDT_COLLECTION_NAME, CSVC_COLLECTION_NAME]
    data_files = [FINAL_LAPTOP_DATA, FINAL_CSBH_DATA, FINAL_CSDT_DATA, FINAL_CSVC_DATA]

    try:
        laptop_vector_store = chroma_vector_store.load_collection(LAPTOP_COLLECTION_NAME)
    
    except Exception as e:
        print(f"Collection not found, initializing new collections: {e}")
        logger.info(f"Collection not found, initializing new collections., error: {e}")
        with open(FINAL_LAPTOP_DATA, "r", encoding="utf-8") as f:
            laptop_text = f.read()
        laptop_vector_store = chroma_vector_store.initialize_collection(
            collection_name= LAPTOP_COLLECTION_NAME,
            text = laptop_text
        )
    
    similarity_results = chroma_vector_store.similarity_search(query, laptop_vector_store)
    results = chroma_vector_store.hybrid_search(query, laptop_vector_store)
    bm25_results = results['bm25_retriever']

    return similarity_results

if __name__ == "__main__":
    query = "CPU của laptop Dell Inspiron 14 5441 là gì?"
    results = retrieve(query)
    print(results)




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

    