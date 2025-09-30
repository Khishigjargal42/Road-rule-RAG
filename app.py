# app.py — Serve index.html + Road Rule RAG API
import os
import base64
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# -------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------------
# Retrieval & generation funcs (from main.py)
# -------------------------------------------------------------------
from main import (
    get_retriever,
    is_road_rule_query,
    synthesize_with_openai,
    extractive_fallback,
)

# -------------------------------------------------------------------
# Ollama fallback (optional)
# -------------------------------------------------------------------
try:
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama deps not found. OpenAI + extractive fallback only.")

# -------------------------------------------------------------------
# Branding / assets
# -------------------------------------------------------------------
DEFAULT_LOGO = os.getenv("LOGO_PATH", "./static/roadrule-logo.png")

def _logo_data_uri() -> str:
    if not os.path.exists(DEFAULT_LOGO):
        return ""
    try:
        with open(DEFAULT_LOGO, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        ext = DEFAULT_LOGO.lower().split(".")[-1]
        mime = "image/png" if ext == "png" else "image/jpeg"
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(title="Road Rule Assistant")

# Static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Location of UI file
BASE_DIR = Path(__file__).parent
INDEX_PATH = BASE_DIR / "index.html"

class AskBody(BaseModel):
    question: str
    top_k: int | None = None
    search_type: str | None = None  # "mmr" | "similarity"

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _collect_sources(docs) -> list:
    out = []
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        out.append({
            "tag": f"s{i}",
            "section": md.get("section", ""),
            "source": md.get("source") or md.get("title") or "unknown",
            "page": md.get("page"),
        })
    return out

def _answer_with_ollama(question: str, docs) -> str:
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("Ollama not available")
    ctx_blocks = []
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        section = md.get("section") or ""
        source = md.get("source") or ""
        header = f"[s{i}] {section} — {source}".strip(" —")
        ctx_blocks.append(f"{header}\n{d.page_content.strip()}")
    context = "\n\n".join(ctx_blocks)

    system = (
        "Та Замын хөдөлгөөний дүрмийн туслах. CONTEXT-оос л баримт ав. "
        "Хүрэлцэхгүй бол 'Мэдээлэл хүрэлцэхгүй' гэж хэл. Монгол хэлээр, товч ойлгомжтой бич. "
        "Эцэст нь [s1], [s2]… маягаар ишлэл өг."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Асуулт: {question}\n\nCONTEXT:\n{context}"),
    ])
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
        temperature=0.2,
        timeout=120,
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context})

def generate_answer(question: str, docs) -> tuple[str, str]:
    try:
        ans = synthesize_with_openai(question, docs)
        return ans, "OpenAI synthesis"
    except Exception as e_openai:
        print(f"OpenAI failed: {e_openai}")
        if OLLAMA_AVAILABLE:
            try:
                ans = _answer_with_ollama(question, docs)
                return ans, "Ollama fallback"
            except Exception as e_ollama:
                print(f"Ollama failed: {e_ollama}")
        try:
            ans = extractive_fallback(question, docs)
            return ans, f"Extractive fallback (OpenAI error: {str(e_openai)[:80]})"
        except Exception as e_extract:
            return (f"Гурван арга хоёул алдаа: OpenAI={e_openai}, Extractive={e_extract}", "Error")

# -------------------------------------------------------------------
# Fallback HTML (if index.html is missing)
# -------------------------------------------------------------------
def get_fallback_html() -> str:
    logo = _logo_data_uri()
    logo_img = (
        f"<img src='{logo}' alt='Road Rule logo' style='height:28px'/>"
        if logo else "<div style='width:10px;height:10px;border-radius:50%;background:#2563eb'></div>"
    )
    return f"""<!doctype html>
<html lang='mn'>
<head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Road Rule Assistant</title>
</head>
<body>
  <header>{logo_img}<div>Road Rule Assistant (Fallback UI)</div></header>
  <main>
    <p>Таны үндсэн <code>index.html</code> олдсонгүй. Root-д <code>index.html</code> байрлуулна уу.</p>
  </main>
</body>
</html>"""

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    if INDEX_PATH.exists():
        return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(get_fallback_html())

@app.post("/ask")
def ask(body: AskBody):
    try:
        q = (body.question or "").strip()
        if not q:
            return JSONResponse({"answer": "Асуулт хоосон байна.", "sources": [], "method": "validation"})

        if not is_road_rule_query(q):
            return JSONResponse({
                "answer": "Зөвхөн замын хөдөлгөөний дүрэмтэй холбоотой асуултад хариулна.",
                "sources": [],
                "method": "domain guard",
            })

        retriever = get_retriever(k=body.top_k or 5, search_type=body.search_type or "mmr")
        docs = retriever.invoke(q)
        if not docs:
            return JSONResponse({"answer": "Мэдээлэл олдсонгүй. Индексээ шалгана уу.", "sources": [], "method": "no retrieval"})

        sources = _collect_sources(docs)
        answer, method = generate_answer(q, docs)
        return JSONResponse({"answer": answer, "sources": sources, "method": method})
    except Exception as e:
        return JSONResponse({"answer": f"Серверийн алдаа гарлаа: {str(e)}", "sources": [], "method": "server error"})

@app.get("/health")
def health_check():
    info = {
        "status": "healthy",
        "openai_available": bool(os.getenv("OPENAI_API_KEY")),
        "ollama_available": OLLAMA_AVAILABLE,
        "vector_store": os.path.exists(os.getenv("PERSIST_DIR", "./chroma_db")),
    }
    return JSONResponse(info)

# -------------------------------------------------------------------
# Entry
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("🌐 http://localhost:8001")
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8001")), reload=True)
