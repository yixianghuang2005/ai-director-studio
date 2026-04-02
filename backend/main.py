import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS：允許前端（Vercel）跨域呼叫
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 部署後可改成你的 Vercel 網址
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_KEY_PAID = os.getenv("GEMINI_API_KEY_PAID", GEMINI_KEY)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# ── 文字生成（分鏡腳本）──
@app.post("/api/gemini/text")
async def proxy_text(request: Request):
    body = await request.json()
    key = GEMINI_KEY_PAID or GEMINI_KEY
    url = f"{GEMINI_BASE}/gemini-2.5-flash:generateContent?key={key}"
    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(url, json=body)
    return JSONResponse(res.json(), status_code=res.status_code)

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

# ── Imagen 3 生圖（9:16 原生支援）──
@app.post("/api/imagen3/generate")
async def proxy_imagen3(request: Request):
    body = await request.json()
    key = GEMINI_KEY_PAID or GEMINI_KEY
    url = f"{GEMINI_BASE}/imagen-3.0-generate-002:predict?key={key}"
    prompt = body.get("prompt", "")
    imagen_body = {
        "instances": [{"prompt": prompt}],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "9:16",
            "safetySetting": "block_only_high"
        }
    }
    async with httpx.AsyncClient(timeout=120) as client:
        res = await client.post(url, json=imagen_body)
    data = res.json()
    # 轉換成前端熟悉的格式
    if "predictions" in data and data["predictions"]:
        b64 = data["predictions"][0].get("bytesBase64Encoded", "")
        if b64:
            return JSONResponse({
                "candidates": [{
                    "content": {
                        "parts": [{"inlineData": {"mimeType": "image/png", "data": b64}}]
                    }
                }]
            })
    return JSONResponse(data, status_code=res.status_code)

# ── 健康檢查 ──
@app.get("/")
def health():
    return {"status": "ok", "service": "超級導演分鏡系統 API"}