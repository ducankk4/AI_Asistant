from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

EMBEDDING_MODEL_VN = "text-embedding-3-large"
GOOGLE_EMBEDDING_MODEL = "gemini-embedding-001"
ROUTING_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
RERANKING_MODEL = ""
RESPONSE_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
REWRITE_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
ANALYSIS_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

class DataPaths:
    FINAL_LAPTOP_DATA = r"src\crawl_data\final_data\laptop.txt"
    FINAL_CSBH_DATA = r"src\crawl_data\final_data\csbh.txt"
    FINAL_CSDT_DATA = r"src\crawl_data\final_data\csdt.txt"
    FINAL_CSVC_DATA = r"src\crawl_data\final_data\csvc.txt"

class RAG_config:
    top_k_result: int = 3
    chunk_size: int = 512
    chunk_overlap: int = 100
    persist_directory: str = r"src\vector_store\chroma_collections"
class CollectionNames:
    LAPTOP_COLLECTION_NAME: str = "laptop_collection"
    CSBH_COLLECTION_NAME: str = "csbh_collection"
    CSDT_COLLECTION_NAME: str = "csdt_collection"
    CSVC_COLLECTION_NAME: str = "csvc_collection"

rag_config = RAG_config()
