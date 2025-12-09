REWRITE_PROMPT = """Hãy viết lại câu hỏi sau cho rõ ràng và đầy đủ dựa trên lịch sử hội thoại bên dưới để sử dụng câu hỏi cho việc truy vấn trong 
vectordatabase
câu hỏi cần viết lại :{query}
lịch sử hội thoại: {chat_history}"""


ROUTING_PROMPT = """hãy phân loại câu hỏi sau theo các collections(được mô tả chi tiết ở bên dưới hãy đọc kĩ mô tả để đưa ra phân loại chính xác) có sẵn trong vectordatabase: laptop, csbh, csdt, csvc. Chỉ trả lời tên collection và lý do chọn collection đó.
câu hỏi: {query}
mô tả các collections:
***laptop: chứa thông tin, đặc điểm nổi bật về các loại laptop, cấu hình, đánh giá.
***csbh: chứa thông tin về chính sách bảo hành, quy trình bảo hành, các trung tâm bảo hành.
***csdt: chứa thông tin về các sự cố thường gặp, cách khắc phục sự cố kỹ thuật.
***csvc: chứa thông tin về dịch vụ vận chuyển, cách thức vận chuyển đơn hàng của cửa hàng.
ví dụ:
câu hỏi: "tháng trước tôi có mua 1 cái laptop dell của cửa hàng bạn giờ nó đang bị lỗi màn hình tôi cần làm gì "
trả lời: collection needed: csbh; reasoning: câu hỏi liên quan đến việc xử lý khi sản phẩm bị lỗi trong thời gian bảo hành, nên cần truy cập vào collection csbh để tìm thông tin về chính sách và quy trình bảo hành.
"""

RESPONSE_PROMPT = """ bạn là 1 trợ lý ai đẳng cấp hãy trả lời câu hỏi của người dùng dựa trên nội dung sau nếu nội dung không liên quan thì trả lời là tôi không biết
query: {query}
context: {context}
"""
