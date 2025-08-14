import os
import sys
import asyncio
import json
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import httpx
from crawl4ai import AsyncWebCrawler
import nest_asyncio

# Setup event loop untuk Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

nest_asyncio.apply()

# Ollama config
OLLAMA_HOST = "http://192.168.100.3:11434"
OLLAMA_MODEL = "llama3.2:latest"

# Template setup
templates = Jinja2Templates(directory="templates")

class ArticleSummary(BaseModel):
    judul: str
    tanggal: str
    penulis: str
    ringkasan: str

async def crawl_url(url: str) -> str:
    """Ambil isi artikel dari URL menggunakan crawler atau fallback HTTP GET."""
    try:
        async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
            result = await crawler.arun(url=url, word_count_threshold=1)
            return result.markdown if result.markdown else result.cleaned_html
    except Exception:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url)
            return response.text

async def summarize_with_ollama(text: str) -> ArticleSummary:
    if len(text) > 10000:
        text = text[:10000]

    # Step 1: Ambil ringkasan mentah
    summary_prompt = f"""
    Buat ringkasan 5 kalimat yang jelas, padat, dan mudah dipahami dari artikel berikut.
    Sertakan poin-poin penting yang dibahas.

    Artikel:
    {text}
    """

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": summary_prompt, "stream": False}
        )
        resp.raise_for_status()
        raw_summary = resp.json().get("response", "").strip()

    print("\n=== RAW SUMMARY OLLAMA ===")
    print(raw_summary)
    print("==========================\n")

    # Step 2: Buat ArticleSummary langsung
    return ArticleSummary(
        judul="tidak tersedia",
        tanggal="tidak tersedia",
        penulis="tidak tersedia",
        ringkasan=raw_summary if raw_summary else "tidak tersedia"
    )

app = FastAPI()

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/summarize")
async def summarize(url: str = Form(...)):
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        article_text = await crawl_url(url)

        if not article_text or len(article_text.strip()) < 100:
            return JSONResponse({
                "success": False,
                "error": "Artikel terlalu pendek atau tidak dapat diambil."
            })

        summary = await summarize_with_ollama(article_text)

        return JSONResponse({
            "success": True,
            "data": summary.model_dump(),
            "url": url
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"Gagal memproses artikel: {str(e)}"
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
