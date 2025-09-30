#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
import hashlib
from pathlib import Path
from docx import Document   # pip install python-docx

CHAPTER_WORDS = r"(Нэг|Хоёр|Гурав|Дөрөв|Тав|Зургаа|Долоо|Найм|Ес|Арав)"
chapter_re = re.compile(rf"^{CHAPTER_WORDS}\.\s+(.+)$")
article_re = re.compile(r"^(\d+(?:\.\d+)+)\.\s+(.*)$")
penalty_re = re.compile(r"(торгуу|шийтгэл)|₮|\b(мянга|сая)\s*төгрөг", re.IGNORECASE)

def make_block_id(chapter: str, article: str, text: str) -> str:
    h = hashlib.sha1()
    key = f"{chapter or ''}||{article or ''}||{text.strip()}"
    h.update(key.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]

def parse_docx_to_records(docx_path: Path, source_name: str):
    doc = Document(str(docx_path))
    chapter = None
    records = []
    buffer_article_id, buffer_text_parts = None, []

    def flush_buffer():
        nonlocal buffer_article_id, buffer_text_parts, chapter, records
        if buffer_article_id is None:
            return
        text = " ".join([t.strip() for t in buffer_text_parts]).strip()
        rec_type = "definition" if ("“" in text and "”" in text) else "rule"
        if penalty_re.search(text):
            rec_type = "penalty" if rec_type != "definition" else "definition"

        rec = {
            "chapter": chapter,
            "article": buffer_article_id,
            "clause": buffer_article_id,
            "type": rec_type,
            "text": text,
            "source_file": source_name,
            "block_id": make_block_id(chapter, buffer_article_id, text)
        }

        mterm = re.search(r"“([^”]+)”", text)
        if mterm:
            rec["term"] = mterm.group(1)

        records.append(rec)
        buffer_article_id, buffer_text_parts = None, []

    for para in doc.paragraphs:
        ln = para.text.strip()
        if not ln:
            continue
        mch = chapter_re.match(ln)
        if mch:
            flush_buffer()
            chapter = f"{mch.group(0)}"
            continue
        ma = article_re.match(ln)
        if ma:
            flush_buffer()
            buffer_article_id = ma.group(1)
            buffer_text_parts = [ma.group(2)]
        else:
            if buffer_article_id is not None:
                buffer_text_parts.append(ln)
    flush_buffer()
    return records

def main():
    ap = argparse.ArgumentParser(description="Convert DOCX -> structured JSON for RAG")
    ap.add_argument("--input", "-i", required=True, help="Path to .docx file")
    ap.add_argument("--output", "-o", required=True, help="Path to output .json")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.stderr.write(f"Input not found: {in_path}\n")
        sys.exit(1)

    records = parse_docx_to_records(in_path, in_path.name)
    out_path = Path(args.output)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records -> {out_path}")

if __name__ == "__main__":
    main()
