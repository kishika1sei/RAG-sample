# app/services/pdf_utils.py
import os
from typing import List, Dict
from flask import current_app
from pypdf import PdfReader
from .llm_utils import embed_texts


# 許可するファイル形式を設定
ALLOWED_EXTS = {".pdf", ".txt",".md",".markdown"}

# returnに書いてるany関数の部分がわからない
def is_allowed_ext(filename: str) -> bool:
    """許可拡張子かどうか（小文字化して判定）"""
    name = filename.lower()
    return any(name.endswith(ext) for ext in ALLOWED_EXTS)

# ===== 読み取り系 =====
def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)

def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_md(path: str) -> str:
    """
    Markdownはまずはテキストとして読み込むだけ（MVP）。
    必要ならfront-matter除去や見出しごとの分割を後で拡張。
    """
    return _read_txt(path)

def read_preview(path: str, limit: int = 3000) -> str:
    """
    画面表示・RAG要約用に軽量プレビューを返す。
    拡張子に応じて読み分ける。
    """
    low = path.lower()
    if low.endswith(".pdf"):
        text = _read_pdf(path)
    elif low.endswith((".md", ".markdown")):
        text = _read_md(path)
    else:
        text = _read_txt(path)
    return text[:limit]

# ===== 分割・インデックス =====
# 日本語の形態素って何？
def _split(text: str, size: int = 800, overlap: int = 120) -> List[str]:
    """
    シンプルな固定長スライス（日本語の形態素分割は後で拡張でOK）
    """
    out, i = [], 0
    n = len(text)
    while i < n:
        chunk = text[i:i + size].strip()
        if chunk:
            out.append(chunk)
        i += size - overlap
    return out

def ingest_local_dir() -> int:
    """
    PDF_DIR を走査→ {pdf,txt,md,markdown} のみ取り込み →
    （PDFはページ単位で）チャンク化→埋め込み→FAISS保存
    """
    pdf_dir = current_app.config["PDF_DIR"]
    os.makedirs(pdf_dir, exist_ok=True)

    texts: List[str] = []
    metas: List[Dict] = []

    for name in sorted(os.listdir(pdf_dir)):
        path = os.path.join(pdf_dir, name)
        if not os.path.isfile(path):
            continue
        if not is_allowed_ext(name):
            continue

        low = name.lower()
        if low.endswith(".pdf"):
            # --- PDFはページごとに処理して page / total_pages をメタへ入れる ---
            reader = PdfReader(path)
            total_pages = len(reader.pages)
            for page_no, page in enumerate(reader.pages, start=1):  # 1始まり（UIに優しい）
                page_text = page.extract_text() or ""
                for j, chunk in enumerate(_split(page_text)):
                    texts.append(chunk)
                    metas.append({
                        "doc": name,
                        "path": path,
                        "chunk_id": f"{page_no}-{j}",
                        "page": page_no,
                        "total_pages": total_pages,
                    })
        elif low.endswith((".md", ".markdown")):
            raw = _read_md(path)
            for i, chunk in enumerate(_split(raw)):
                texts.append(chunk)
                metas.append({
                    "doc": name, "path": path, "chunk_id": i,
                    # テキスト系はページ概念が無いので total_pages は None
                    "total_pages": None,
                })
        else:
            raw = _read_txt(path)
            for i, chunk in enumerate(_split(raw)):
                texts.append(chunk)
                metas.append({
                    "doc": name, "path": path, "chunk_id": i,
                    "total_pages": None,
                })

    if not texts:
        return 0

    vecs = embed_texts(texts, model=current_app.config["EMBED_MODEL"])
    from .vectorstore import faiss_save
    faiss_save(vecs, metas)
    # 取り込んだファイル数（doc単位のユニーク数）
    return len(set(m["doc"] for m in metas))


def page_count_pdf(path: str) -> int:
    """PDFのページ数を返す。失敗時は0"""
    try:
        reader = PdfReader(path)
        return len(reader.pages)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return 0