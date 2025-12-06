from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

EMBEDDING_MODEL_VN = "text-embedding-3-large"
GOOGLE_EMBEDDING_MODEL = "gemini-embedding-001"

FINAL_LAPTOP_DATA = r"crawl_data\final_data\laptop.txt"
FINAL_CSBH_DATA = r"crawl_data\final_data\csbh.txt"
FINAL_CSDT_DATA = r"crawl_data\final_data\csdt.txt"
FINAL_CSVC_DATA = r"crawl_data\final_data\csvc.txt"

class RAG_config:
    top_k_result: int = 3
    chunk_size: int = 1000
    chunk_overlap: int = 150
    persist_directory: str = r"vector_store\chroma_collections"

LAPTOP_COLLECTION_NAME: str = "laptop_collection"
CSBH_COLLECTION_NAME: str = "csbh_collection"
CSDT_COLLECTION_NAME: str = "csdt_collection"
CSVC_COLLECTION_NAME: str = "csvc_collection"

rag_config = RAG_config()
