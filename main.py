import os
import sys
import asyncio
import json
import re
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import httpx
from crawl4ai import AsyncWebCrawler
import nest_asyncio
from bs4 import BeautifulSoup

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
nest_asyncio.apply()

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found")

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

class ArticleSummary(BaseModel):
    judul: str
    tanggal: str
    penulis: str
    ringkasan: str

STYLE_CONTEXTS = {
    "formal": {
        "name": "Formal",
        "description": "Bahasa resmi dan akademis",
        "prompt_addition": "Gunakan bahasa formal yang sesuai untuk dokumen resmi, akademis, atau profesional. Hindari kontraksi dan gunakan struktur kalimat yang lengkap dan baku. Gunakan terminologi yang tepat dan presisi."
    },
    "casual": {
        "name": "Kasual", 
        "description": "Bahasa santai dan mudah dipahami",
        "prompt_addition": "Gunakan bahasa yang santai, ramah, dan mudah dipahami seperti berbicara dengan teman. Boleh menggunakan kontraksi dan kata-kata sehari-hari. Buatlah seperti sedang bercerita kepada teman."
    },
    "professional": {
        "name": "Profesional",
        "description": "Bahasa bisnis dan teknis", 
        "prompt_addition": "Gunakan bahasa profesional yang cocok untuk lingkungan bisnis atau kerja. Fokus pada aspek praktis, dampak, dan implikasi bisnis. Gunakan terminologi industri yang relevan."
    },
    "journalistic": {
        "name": "Jurnalistik",
        "description": "Gaya berita dan informatif",
        "prompt_addition": "Gunakan gaya penulisan jurnalistik yang objektif, informatif, dan menarik. Fokus pada fakta-fakta penting, siapa, apa, kapan, di mana, mengapa, dan bagaimana. Gunakan lead yang kuat."
    },
    "educational": {
        "name": "Edukatif",
        "description": "Gaya pengajaran dan pembelajaran",
        "prompt_addition": "Gunakan pendekatan edukatif seperti seorang guru atau dosen yang menjelaskan materi kepada siswa. Berikan konteks, penjelasan istilah yang mungkin tidak familiar, dan hubungkan dengan konsep yang lebih besar."
    },
    "simple": {
        "name": "Sederhana", 
        "description": "Bahasa yang sangat mudah dipahami",
        "prompt_addition": "Gunakan bahasa yang sangat sederhana dan mudah dipahami oleh semua kalangan. Hindari jargon teknis, gunakan kata-kata sehari-hari, dan jelaskan konsep rumit dengan analogi sederhana."
    }
}

def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    main_content = soup.find("article") or soup.find("main") or soup.find("body") or soup
    return main_content.get_text(separator="\n", strip=True)

async def crawl_url(url: str) -> str:
    try:
        async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
            result = await crawler.arun(url=url, word_count_threshold=1)
            if getattr(result, "markdown", None):
                return result.markdown
            if getattr(result, "cleaned_html", None):
                return clean_html(result.cleaned_html)
    except Exception:
        pass

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return clean_html(resp.text)

def safe_extract_json(raw: str) -> dict:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end+1])
        raise

async def summarize_with_gemini(text: str, style: str = "casual") -> ArticleSummary:
    if len(text) > 10000:
        text = text[:10000]

    style_config = STYLE_CONTEXTS.get(style, STYLE_CONTEXTS["casual"])
    style_instruction = style_config["prompt_addition"]

    prompt = f"""
    Analisis artikel berikut dan berikan ringkasan dalam format JSON yang valid dengan struktur:
    {{
        "judul": "judul artikel",
        "tanggal": "tanggal publikasi atau 'tidak tersedia'",
        "penulis": "nama penulis atau 'tidak tersedia'", 
        "ringkasan": "ringkasan artikel"
    }}

    INSTRUKSI GAYA PENULISAN:
    {style_instruction}

    INSTRUKSI RINGKASAN:
    Pelajari isi artikel yang sudah diberikan lalu berikan kepada saya penjelasan yang singkat, jelas, dan padat dari artikel tersebut. Buat ringkasan dalam bentuk paragraf sepanjang 5 kalimat dan pada setiap kalimat saya berharap Anda mencantumkan poin penting dari artikel yang telah dibaca. 

    Berikan dengan kata-kata Anda sendiri yang dapat mudah dipahami. Intinya bukan membahas tentang apa yang ada di dalam artikel, namun artikel tersebut membahas apa dan Anda menjelaskannya kepada saya, layaknya seorang dosen yang membaca buku lalu memberikan penjelasan terhadap mahasiswanya.

    PENTING: Gunakan bahasa Indonesia yang sesuai dengan gaya "{style_config['name']}" yang telah dipilih.
    
    Artikel:
    {text}
    """

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json=payload
        )
        resp.raise_for_status()
        data = resp.json()

    raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    parsed_json = safe_extract_json(raw_text)

    return ArticleSummary(**parsed_json)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("gemini.html", {"request": request})

@app.post("/summarize")
async def summarize(url: str = Form(...), style: str = Form("casual")):
    try:
        if style not in STYLE_CONTEXTS:
            return JSONResponse({
                "success": False,
                "error": f"Gaya '{style}' tidak valid. Pilih dari: {', '.join(STYLE_CONTEXTS.keys())}"
            })

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        article_text = await crawl_url(url)
        if not article_text or len(article_text.strip()) < 100:
            return JSONResponse({
                "success": False,
                "error": "Artikel terlalu pendek atau tidak dapat diambil. Pastikan URL artikel valid."
            })

        summary = await summarize_with_gemini(article_text, style)

        return JSONResponse({
            "success": True,
            "data": summary.model_dump(),
            "url": url,
            "style": style,
            "style_info": STYLE_CONTEXTS[style]
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"Gagal memproses artikel: {str(e)}"
        })

@app.get("/styles")
async def get_styles():
    return JSONResponse({
        "success": True,
        "styles": STYLE_CONTEXTS
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)