import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_KEY_PAID = os.getenv("GEMINI_API_KEY_PAID", GEMINI_KEY)
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

TEXT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

# ── 文字生成（自動 fallback 三個模型）──
@app.post("/api/gemini/text")
async def proxy_text(request: Request):
    body = await request.json()
    key = GEMINI_KEY_PAID or GEMINI_KEY
    last_res = None
    for model in TEXT_MODELS:
        url = f"{GEMINI_BASE}/{model}:generateContent?key={key}"
        async with httpx.AsyncClient(timeout=60) as client:
            res = await client.post(url, json=body)
        last_res = res
        if res.status_code == 200:
            return JSONResponse(res.json(), status_code=200)
        if res.status_code not in [503, 429]:
            return JSONResponse(res.json(), status_code=res.status_code)
    return JSONResponse(last_res.json(), status_code=last_res.status_code)

# ── 圖片生成（主模型）──
@app.post("/api/gemini/image")
async def proxy_image(request: Request):
    body = await request.json()
    key = GEMINI_KEY_PAID or GEMINI_KEY
    url = f"{GEMINI_BASE}/gemini-2.5-flash:generateContent?key={key}"
    async with httpx.AsyncClient(timeout=120) as client:
        res = await client.post(url, json=body)
    return JSONResponse(res.json(), status_code=res.status_code)

# ── 圖片生成（備用模型）──
@app.post("/api/gemini/image-fallback")
async def proxy_image_fallback(request: Request):
    body = await request.json()
    key = GEMINI_KEY_PAID or GEMINI_KEY
    url = f"{GEMINI_BASE}/gemini-2.5-flash-image:generateContent?key={key}"
    async with httpx.AsyncClient(timeout=120) as client:
        res = await client.post(url, json=body)
    return JSONResponse(res.json(), status_code=res.status_code)

# ── 健康檢查 ──
@app.get("/")
def health():
    return {"status": "ok", "service": "超級導演分鏡系統 API"}