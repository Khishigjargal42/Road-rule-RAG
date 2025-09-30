# test_query.py  — интерактив асууж, "exit" гэж бичихэд дуусна
import os
import sys

from langchain_chroma import Chroma

# HuggingFaceEmbeddings импортыг шинэ пакетнаас оролдоод, бүтээгүй бол хуучнаар fallback хийнэ
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # pip install -U langchain-huggingface
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # pip install -U langchain-community sentence-transformers

# --------------------
# Тохиргоо
# --------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K = 5

# Проектын хавтаснаасаа харьцангуйгаар Chroma_db-г заана (зураг дээрх бүтэцтэй тааруулсан)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "Chroma_db")  # өөрчлөх шаардлагагүй, хавтас чинь ингэж нэрлэгдсэн

COLLECTION = "road_rules"

# --------------------
# Инициализац
# --------------------
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

db = Chroma(
    collection_name=COLLECTION,
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)

# DB доторх векторын тоо (шалгах)
try:
    n = len(db.get(include=[]).get("ids", []))
    print(f"[DB] persist='{PERSIST_DIR}', collection='{COLLECTION}', vectors={n}")
except Exception:
    pass

print("Interactive RAG 🔎 — асуултаа бичээд Enter дар. Дуусгах бол 'exit' (эсвэл 'quit', 'q').")

# --------------------
# Интерактив цикл
# --------------------
while True:
    try:
        q = input("\nQ> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    if not q:
        continue

    if q.lower() in ("exit", "quit", "q"):
        print("Bye!")
        break

    results = db.similarity_search_with_score(q, k=K)

    if not results:
        print("→ Илэрц олдсонгүй.")
        continue

    print(f"[Асуулт] {q}  (k={K})\n")
    for i, (doc, score) in enumerate(results, 1):
        meta = {k: v for k, v in doc.metadata.items() if v is not None}
        snippet = doc.page_content.replace("\n", " ")
        if len(snippet) > 250:
            snippet = snippet[:250] + " ..."
        print(f"--- Хариулт {i} (score={score:.4f}) ---")
        print("Metadata:", meta)
        print("Текст:", snippet)

