from langchain_core.documents import Document
from langchain_chroma.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from typing import List
from src.embedding.google_embed import GoogleEmbedding
from src.logger import logger
from src.config import rag_config,collection_names, data_paths, GOOGLE_API_KEY, GOOGLE_EMBEDDING_MODEL
import chromadb
import os

class ChromaVectorStore:
    def __init__(self):
        self.embedding_function = GoogleEmbedding(GOOGLE_API_KEY, GOOGLE_EMBEDDING_MODEL)
        self.rag_config = rag_config
        self.collection_names = collection_names
        self.data_paths = data_paths
        self.client = chromadb.PersistentClient(
            path= self.rag_config.persist_directory
        )
        self.laptop_collection = None
        self.csbh_collection = None
        self.csdt_collection = None
        self.csvc_collection = None

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

    async def hybrid_search(self, query: str, vector_store: Chroma) -> List[Document] :

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
        results = await ensemble_retriever.ainvoke(query)
        return results
    
    def initialize_collections(self):
        collections = self.client.list_collections()
        collection_names = [col.name for col in collections]
        logger.info(f"existing {len(collections)} collections: {[col.name for col in collections]}")
        
        try:
            # Laptop collection
            if self.collection_names.LAPTOP_COLLECTION_NAME in collection_names:
                self.laptop_collection = self.load_collection(collection_name=self.collection_names.LAPTOP_COLLECTION_NAME)
            else:
                with open(self.data_paths.FINAL_LAPTOP_DATA, "r", encoding="utf-8") as f:
                    text = f.read()
                logger.info(f"initializing collection: {self.collection_names.LAPTOP_COLLECTION_NAME}")
                self.laptop_collection = self.create_collection(collection_name=self.collection_names.LAPTOP_COLLECTION_NAME, text=text)
            
            # CSBH collection
            if self.collection_names.CSBH_COLLECTION_NAME in collection_names:
                self.csbh_collection = self.load_collection(collection_name=self.collection_names.CSBH_COLLECTION_NAME)
            else:
                with open(self.data_paths.FINAL_CSBH_DATA, "r", encoding="utf-8") as f:
                    text = f.read()
                logger.info(f"initializing collection: {self.collection_names.CSBH_COLLECTION_NAME}")
                self.csbh_collection = self.create_collection(collection_name=self.collection_names.CSBH_COLLECTION_NAME, text=text)
            
            # CSDT collection
            if self.collection_names.CSDT_COLLECTION_NAME in collection_names:
                self.csdt_collection = self.load_collection(collection_name=self.collection_names.CSDT_COLLECTION_NAME)
            else:
                with open(self.data_paths.FINAL_CSDT_DATA, "r", encoding="utf-8") as f:
                    text = f.read()
                logger.info(f"initializing collection: {self.collection_names.CSDT_COLLECTION_NAME}")
                self.csdt_collection = self.create_collection(collection_name=self.collection_names.CSDT_COLLECTION_NAME, text=text)
            
            # CSVC collection
            if self.collection_names.CSVC_COLLECTION_NAME in collection_names:
                self.csvc_collection = self.load_collection(collection_name=self.collection_names.CSVC_COLLECTION_NAME)
            else:
                with open(self.data_paths.FINAL_CSVC_DATA, "r", encoding="utf-8") as f:
                    text = f.read()
                logger.info(f"initializing collection: {self.collection_names.CSVC_COLLECTION_NAME}")
                self.csvc_collection = self.create_collection(collection_name=self.collection_names.CSVC_COLLECTION_NAME, text=text)
                
        except Exception as e:
            logger.error(f"error in initialize_collections: {e}")


                





      
# if __name__ == "__main__":
#     chromavts = ChromaVectorStore()
#     text = """giá sản phẩm đang được khuyến mãi là: 18.890.000 VNĐ, chi tiết mô tả sản phẩm tham khảo tại đây: https://www.ankhang.vn/laptop-acer-aspire-go-14-ai-ag14-71m-7681-nx.jfwsv.002.html, ĐẶC ĐIỂM NỔI BẬT Laptop Acer Aspire Go 14 AI AG14-71M-7681 NX.JFWSV.002 - Laptop AI mỏng nhẹ giá tốt 2025 Laptop Acer Aspire Go 14 AI AG14-71M-7681 NX.JFWSV.002 là chiếc laptop AI mỏng nhẹ mới của Acer năm 2025, hướng đến người dùng cần một thiết bị di động nhưng vẫn đảm bảo hiệu năng. Nổi bật với chip Intel Core Ultra 7 155H tích hợp NPU AI Boost hiện đại, cùng màn hình 14 inch FHD+ chuẩn màu và trọng lượng chỉ 1.5kg, chiếc laptop này mang đến sự cân bằng hoàn hảo giữa hiệu năng và tính di động. Trong tầm giá chỉ từ 16-20 triệu đồng, Acer Aspire Go 14 AI AG14-71M-7681 là sự lựa chọn phù hợp cho học sinh, sinh viên và dân văn phòng.
#  Hiệu năng AI mạnh mẽ với CPU Intel Core Ultra 7 155H Trang bị bộ vi xử lý Intel Core Ultra 7 155H với 16 nhân, 22 luồng cùng xung nhịp tối đa lên đến 4.8GHz, Acer Aspire Go 14 AI AG14-71M-7681 NX.JFWSV.002 mang đến hiệu năng ấn tượng trong tầm giá, đủ sức xử lý đa nhiệm nặng từ học tập, văn phòng đến chỉnh sửa hình ảnh, video cơ bản. Điểm khác biệt lớn nằm ở NPU Intel AI Boost tích hợp, cho phép máy chạy mượt các ứng dụng hỗ trợ trí tuệ nhân tạo, dịch ngôn ngữ, tối ưu hình ảnh hay khai thác trọn vẹn Windows Copilot mà không tiêu tốn quá nhiều năng lượng.
#  Trong phân khúc giá từ 16 - 20 triệu đồng, hiếm có model nào vừa sở hữu CPU mạnh vừa tích hợp AI NPU như Aspire Go 14 AI. Đây chính là lợi thế giúp chiếc laptop này bắt kịp xu hướng AI PC 2025, mang lại giá trị lâu dài hơn so với những mẫu Core i5/i7 đời trước vốn chưa có khả năng xử lý AI chuyên biệt.
#  Bộ nhớ dung lượng lớn, lưu trữ tốc độ cao Laptop Acer Aspire Go 14 AI AG14-71M-7681 NX.JFWSV.002 được trang bị sẵn 16GB RAM DDR5 5600MHz cùng ổ cứng 512GB SSD PCIe Gen4 NVMe, mang đến khả năng đa nhiệm mượt mà, tốc độ phản hồi ứng dụng nhanh và không gian đủ rộng để cài đặt phần mềm, lưu tài liệu, dữ liệu học tập hay công việc thường ngày. Đặc biệt, máy hỗ trợ nâng cấp RAM tối đa 96GB – một con số hiếm gặp trên laptop mỏng nhẹ. Điều này không chỉ giúp người dùng yên tâm sử dụng lâu dài mà còn mở ra khả năng mở rộng cho nhu cầu chuyên sâu hơn trong tương lai.
#  Màn hình FHD+ 14 inch sắc nét, tối ưu cho học tập và công việc Với màn hình 14 inch FHD+ (1920x1200) IPS tỷ lệ 16:10, Acer Aspire Go 14 AI AG14-71M-7681 NX.JFWSV.002 không chỉ gây ấn tượng với chất lượng hình ảnh rõ nét cùng góc nhìn rộng lên đến 178 độ mà còn mang đến không gian hiển thị rộng rãi hơn cho đọc tài liệu, viết báo cáo hay lập trình.
#  Khả năng tái tạo màu cũng rất ấn tượng khi đạt 100% sRGB, vượt trội so với những mẫu laptop phổ thông chỉ dừng ở mức 45% NTSC. Điều này giúp màu sắc hiển thị rực rỡ, trung thực hơn, phù hợp cho các tác vụ chỉnh sửa ảnh, thiết kế đồ họa cơ bản. Công nghệ ComfyView chống chói và BlueLightShield giảm ánh sáng xanh có hại, bảo vệ mắt khi phải sử dụng liên tục. Đây sẽ là sự lựa chọn phù hợp cho học sinh, sinh viên và dân văn phòng khi thường xuyên làm việc với máy tính.
#  Thiết kế mỏng nhẹ chỉ 1.5kg Laptop Acer Aspire Go 14 AI AG14-71M-7681 NX.JFWSV.002 được thiết kế theo phong cách tối giản với tông xám Steel Gray trung tính, tạo cảm giác thanh lịch và dễ hòa nhập trong nhiều môi trường khác nhau, từ giảng đường đến văn phòng làm việc. Đây là ngôn ngữ thiết kế quen thuộc ở phân khúc laptop phổ thông, chú trọng tính thực dụng hơn là sự phô trương.
#  Với trọng lượng chỉ 1.5kg và độ dày chỉ 17.5mm, kết hợp nắp và đáy kim loại để tăng độ chắc chắn, Aspire Go 14 AI mang lại sự cân bằng giữa độ bền bỉ và tính di động. Kích thước 313.6 x 227.3 mm đủ gọn gàng để mang theo mỗi ngày, phù hợp cho người dùng cần một chiếc laptop dễ bỏ vào balo mà không gây nặng nề.
#  Hệ thống kết nối đầy đủ, linh hoạt cho mọi nhu cầu Nếu nhiều laptop mỏng nhẹ phải đánh đổi số cổng để đạt thiết kế gọn gàng, thì Acer Aspire Go 14 AI AG14-71M-7681 NX.JFWSV.002 lại nổi bật với hệ thống kết nối tương đối toàn diện. Máy trang bị USB-C hỗ trợ xuất hình và sạc nhanh, USB-A 3.2 Gen1 cho thiết bị ngoại vi, HDMI 2.1 phục vụ xuất hình ảnh chất lượng cao, kèm theo khe microSD và jack 3.5mm cho nhu cầu cơ bản.
#  Đặc biệt, chiếc laptop Acer Aspire này vẫn giữ lại cổng LAN RJ-45, đảm bảo kết nối mạng có dây ổn định trong môi trường văn phòng hay phòng lab. Song song đó, Wi-Fi 6 và Bluetooth 5.2 mang đến kết nối không dây nhanh chóng, ổn định và giảm thiểu độ trễ khi học tập, làm việc trực tuyến.
#  Trải nghiệm thông minh với Copilot Key và AcerSense Laptop Acer Aspire Go 14 AI AG14-71M-7681 NX.JFWSV.002 mang đến trải nghiệm thân thiện nhờ loạt tính năng thông minh. Nổi bật là Copilot Key, một phím tắt vật lý cho phép bạn truy cập nhanh vào trợ lý AI của Windows 11. Điều này giúp việc tìm kiếm, sáng tạo nội dung và xử lý tác vụ trở nên nhanh chóng hơn. Song song đó, phần mềm độc quyền AcerSense cung cấp một giao diện trực quan để quản lý hiệu năng, pin và cập nhật hệ thống, giúp người dùng phổ thông dễ dàng theo dõi tình trạng thiết bị.
#  Bên cạnh đó, Acer còn tích hợp PurifiedVoice™ và TNR để tối ưu chất lượng âm thanh và hình ảnh trong các cuộc gọi video hay họp trực tuyến. Sự kết hợp này tạo nên một hệ sinh thái trải nghiệm thông minh, dễ dùng và thực sự hữu ích trong cả học tập lẫn công việc hằng ngày.
#  Như vậy, laptop Acer Aspire Go 14 AI AG14-71M-7681 NX.JFWSV.002 là chiếc laptop mỏng nhẹ “đi đúng xu hướng 2025” khi mang đến hiệu năng AI mạnh mẽ, màn hình rõ nét chuẩn màu, thiết kế gọn nhẹ, kết nối đa dạng và đặc biệt là giá dễ tiếp cận. Nếu bạn là sinh viên, nhân viên văn phòng hoặc ai đó cần một chiếc laptop AI để bắt kịp xu hướng làm việc hiện đại, thì Aspire Go 14 AI là một lựa chọn sáng giá.
#  Hiện tại, chiếc laptop này đang có sẵn tại An Khang Computer với mức giá siêu ưu đãi, đi kèm với nhiều phần quà hấp dẫn! Đến ngay showroom hoặc liên hệ với chúng tôi để được sở hữu laptop Acer chính hãng, bảo hành 12 tháng, hỗ trợ trả góp với lãi suất 0%!
#   >> Có thể bạn quan tâm:
#  Laptop Acer Swift Lite 14 AI SFL14-51M-78XZ NX.J1HSV.001 Laptop Acer Swift Go 14 SFG14-41-R5JK NX.KG3SV.002 Laptop Acer Swift Go 14 SFG14-41-R19Z NX.KG3SV.001
# """
#     embeding = chromavts.embedding_function.embed_query(query= text)
#     print(f"Embedding vector length: {len(embeding)}")