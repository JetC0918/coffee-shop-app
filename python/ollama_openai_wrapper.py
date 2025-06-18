from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

@app.post("/v1/chat/completions")
def chat_completion(request_data: ChatCompletionRequest):
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": request_data.model,
                "messages": [msg.dict() for msg in request_data.messages],
                "stream": False,
                "options": {
                    "temperature": request_data.temperature
                }
            }
        )
        response.raise_for_status()
        result = response.json()

        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": 0,
            "model": request_data.model,
            "choices": [{
                "index": 0,
                "message": result["message"],
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    except Exception as e:
        print("🔥 ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
