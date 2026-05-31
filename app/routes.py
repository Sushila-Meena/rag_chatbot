from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.chat import answer

router = APIRouter()


# --- request / response models ---

class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []   # [{"role": "user"|"assistant", "content": "..."}]


class ChatResponse(BaseModel):
    answer: str


# --- routes ---

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        reply = answer(req.question, req.history)
        return ChatResponse(answer=reply)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))