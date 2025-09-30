# main.py — Chroma (retrieval) + OpenAI synthesis for Road Rules
# -----------------------------------------------------------------------------
# - Retrieves top-k chunks from existing Chroma DB (mxbai-embed-large)
# - Synthesizes a coherent answer; falls back to extractive if LLM fails
# -----------------------------------------------------------------------------

import os
import re
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# ============ Config ============
PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "road_rules")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

TOP_K = int(os.getenv("RETRIEVE_K", "5"))
SEARCH_TYPE = os.getenv("SEARCH_TYPE", "mmr")  # "mmr" or "similarity"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))

# Load .env
load_dotenv()

# ============ Domain guard (Road Rules) ============
RR_KEYWORDS = [
    "зам", "дүрэм", "гэрэл", "гэрлэн дохио", "дохио", "тэмдэг", "шугам", "зогс",
    "явган", "гарц", "эргэх", "түр зогсолт", "зөрчих", "торгууль", "дараалал",
    "уулзвар", "нэгдэл", "түргэн тусламж", "осол", "жолоо", "жолооч"
]

def is_road_rule_query(q: str) -> bool:
    s = " ".join((q or "").lower().split())
    if not s:
        return False
    return any(k in s for k in RR_KEYWORDS)

# ============ Retriever ============
def get_retriever(
    k: int = TOP_K,
    search_type: str = SEARCH_TYPE,
    search_kwargs: Optional[dict] = None,
):
    if search_kwargs is None:
        search_kwargs = {"k": k} if search_type == "similarity" else {"k": k, "fetch_k": 20, "lambda_mult": 0.4}
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=os.getenv("OPENAI_API_KEY"))
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

# ============ OpenAI synthesis ============
def synthesize_with_openai(question: str, docs) -> str:
    from openai import OpenAI, RateLimitError, APIError
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    ctx_blocks = []
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        section = md.get("section") or ""
        source = md.get("source") or md.get("title") or ""
        header = f"[{i}] {section} — {source}".strip(" —")
        ctx_blocks.append(f"{header}\n{d.page_content.strip()}")
    context_text = "\n\n".join(ctx_blocks)

    system_msg = (
        "Та туршлагатай Замын хөдөлгөөний дүрмийн зөвлөх. "
        "Зөвхөн өгөгдсөн контекстэд тулгуурлан баримттай, логик дараалалтай, Монгол хэлээр товч ойлгомжтой хариулт бич. "
        "Асуултад шууд хариулж, шаардлагатай бол заалт/жишээ дурд. Таамаг бүү хэл."
    )

    user_msg = textwrap.dedent(f"""
    Асуулт: {question}

    КОНТЕКСТ (retrieved top-{TOP_K}):
    {context_text}
    """)

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=OPENAI_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS,
        )
        return (resp.choices[0].message.content or "").strip()
    except (RateLimitError, APIError):
        raise
    except Exception:
        raise

# ============ Extractive fallback ============
def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?…]|[\n])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def _keyword_score(sentence: str, keywords: List[str]) -> int:
    s = sentence.lower()
    return sum(s.count(k) for k in keywords)

def extractive_fallback(question: str, docs, max_sentences: int = 12) -> str:
    q_words = [w for w in re.findall(r"[а-яa-z0-9\\-]+", question.lower()) if len(w) >= 3]
    q_words = list(dict.fromkeys(q_words))

    candidates: List[tuple[int, str]] = []
    for d in docs:
        for sent in _split_sentences(d.page_content):
            sc = _keyword_score(sent, q_words)
            if sc > 0:
                candidates.append((sc, sent))

    if not candidates:
        big = " ".join(d.page_content.strip() for d in docs[:3])
        return f"Таны асуулттай холбоотойгоор дараах мэдээлэл байна: {big}"

    candidates.sort(key=lambda x: (-x[0], len(x[1])))
    top_sentences = [s for _, s in candidates[:max_sentences]]
    main_text = " ".join(top_sentences[:6])
    additional = " ".join(top_sentences[6:]) if len(top_sentences) > 6 else ""

    result = f"Таны асуулттай холбоотойгоор: {main_text}"
    if additional:
        result += f" Нэмээд: {additional}"
    return result

# ============ CLI test ============
def main():
    print("🟢 Road Rule RAG")
    print(f"   Retrieval: {PERSIST_DIR} / {COLLECTION_NAME} (embed={EMBED_MODEL})")
    print(f"   OpenAI: model={OPENAI_MODEL}, max_tokens={OPENAI_MAX_TOKENS}, temp={OPENAI_TEMPERATURE}\n")

    if not os.path.isdir(PERSIST_DIR):
        print(f"⚠️  '{PERSIST_DIR}' байхгүй байна. Эхлээд индексээ шалгана уу.")
        return

    retriever = get_retriever(k=TOP_K, search_type=SEARCH_TYPE)
    q = "Зогс шугам ба явган гарц давхцах үед хаана зогсох вэ?"
    docs = retriever.invoke(q)
    try:
        ans = synthesize_with_openai(q, docs)
    except Exception as e:
        ans = extractive_fallback(q, docs)
        print("(fallback)", e)
    print(ans)

if __name__ == "__main__":
    main()
