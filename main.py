from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import json
import os
import httpx

load_dotenv()

app = FastAPI(
    title="Aurelius | Life Assistant",
    description="AI-powered personal life assistant",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MEMORY_FILE = Path("memory.json")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
AI_API_KEY = os.getenv("AI_API_KEY")


def load_memory():
    if not MEMORY_FILE.exists():
        memory = {"facts": [], "preferences": [], "goals": [], "events": []}
        save_memory(memory)
        return memory
    with open(MEMORY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory, file, ensure_ascii=False, indent=4)


class MemoryRequest(BaseModel):
    category: str
    content: str


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {
        "name": "Aurelius",
        "status": "online",
        "version": "0.1.0",
        "message": "Aurelius is ready."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/memory")
def get_memory():
    return load_memory()


@app.post("/memory")
def add_memory(memory_request: MemoryRequest):
    memory = load_memory()

    category = memory_request.category

    if category not in memory:
        return {
            "error": "Invalid memory category"
        }

    memory[category].append(memory_request.content)
    save_memory(memory)

    return {
        "status": "saved",
        "category": category,
        "content": memory_request.content
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    memory = load_memory()

    if not AI_API_KEY:
        return {
            "error": "AI_API_KEY is missing"
        }

    system_prompt = f"""
You are Aurelius, a personal life assistant.

Your job is to help the user understand, organize,
and improve their life.

Here is the user's current memory:

{json.dumps(memory, ensure_ascii=False, indent=2)}

Use this memory when it is relevant.
Do not invent personal information that is not present.

User message:
{request.message}
"""

    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openrouter/free",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": request.message
            }
        ]
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload
        )

    if response.status_code != 200:
        return {
            "error": "OpenRouter request failed",
            "status_code": response.status_code,
            "details": response.text
        }

    data = response.json()

    return {
        "reply": data["choices"][0]["message"]["content"]
    }
