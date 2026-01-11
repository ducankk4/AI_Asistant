from fastapi import FastAPI, HTTPException
from src.agent.graphs import FinalGraph, RAGGraph
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# final_graph = FinalGraph()
# # final_executor = final_graph.implement_graph()
rag_graph = RAGGraph()
rag_executor = rag_graph.implement_graph()

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None

@app.get("/")
async def root():
    return {
        "message" : "Welcome to An Khang laptop"
    }

@app.post("/chat", response_model= ChatResponse)
async def chat(request: ChatRequest):
    try:
        results = rag_executor.invoke({
            "query": request.query,
            "need_rewrite": True
        })
        reponse = results["subquery_answers"]
        print(f"response type : {type(reponse)}")
        for r in reponse:
            print()
        
        return ChatResponse(response=reponse)
    except Exception as e:
        raise HTTPException(status_code= 500, detail= str(e))
@app.post("/test", response_model= ChatResponse)
async def test(request: ChatRequest):
    query = request.query
    new_query = query.upper() + "em chao anh duc"
    return ChatResponse(response= new_query)

