from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

# ── CORS 設定 ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Keys（從環境變數讀取）──────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ── 模型清單 ───────────────────────────────────────────────
MODELS = {
    "flux-schnell":  "black-forest-labs/FLUX.1-schnell",
    "sdxl":          "stabilityai/stable-diffusion-xl-base-1.0",
    "realistic":     "SG161222/Realistic_Vision_V6.0_B1_noVAE",
    "epicrealism":   "emilianJR/epiCRealism",
    "openjourney":   "prompthero/openjourney",
    "dreamlike":     "dreamlike-art/dreamlike-photoreal-2.0",
}

# ── Request 格式 ───────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    model_key: str = "flux-schnell"
    seed: int = 42

# ── 生圖 API ───────────────────────────────────────────────
@app.post("/generate")
async def generate_image(req: GenerateRequest):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN not set")

    model_id = MODELS.get(req.model_key)
    if not model_id:
        raise HTTPException(status_code=400, detail="Invalid model_key")

    url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
        "x-wait-for-model": "true",
    }
    body = {
        "inputs": req.prompt,
        "parameters": {
            "seed": req.seed,
            "num_inference_steps": 20,
        }
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        res = await client.post(url, headers=headers, json=body)

    if res.status_code != 200:
        raise HTTPException(status_code=res.status_code, detail=res.text)

    # 回傳圖片 base64
    import base64
    img_b64 = base64.b64encode(res.content).decode("utf-8")
    content_type = res.headers.get("content-type", "image/jpeg")
    return {"image": f"data:{content_type};base64,{img_b64}"}


# ── 健康檢查 ───────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "message": "AI Director Studio Backend"}