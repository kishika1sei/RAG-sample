# app/services/rag.py
from typing import Dict, List
from flask import current_app
from .llm_utils import chat
from .vectorstore import faiss_exists, faiss_search
from .doc_utils import read_preview
from .serp_utils import google_search
import requests
from bs4 import BeautifulSoup

# 初期プロンプト設定
SYS = "あなたは日本語で正確に答えるアシスタントです。根拠に基づき簡潔に回答し、不明な点は正直に『不明』と述べてください。"

def _fetch_text(url: str) -> str:
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n")
    except Exception:
        return ""

def _summarize(contexts: List[str], query: str) -> str:
    user = (
        "以下のコンテキストを根拠に質問へ回答してください。"
        "不足していれば『不明』と記してください。\n\n"
        f"【質問】\n{query}\n\n【コンテキスト】\n" + "\n---\n".join(contexts[:8])
    )
    return chat(
        messages=[{"role":"system","content":SYS},{"role":"user","content":user}],
        model=current_app.config["LLM_MODEL"]
    )

def _doc(query: str) -> Dict:
    if not faiss_exists():
        return {"answer":"インデックスがありません。先に /api/ingest を実行してください。", "sources":[]}
    hits = faiss_search(query, k=5)
    contexts, sources = [], []
    for h in hits[:3]:
        contexts.append(read_preview(h["path"], limit=3000))
        sources.append({"title": h["doc"], "url": None, "score": h["score"], "kind":"doc", "path": h["path"]})
    ans = _summarize(contexts, query) if contexts else "該当ドキュメントが見つかりませんでした。"
    return {"answer": ans, "sources": sources}

def _web(query: str) -> Dict:
    results = google_search(query, pages=1)
    contexts, sources = [], []
    for r in results[:3]:
        txt = _fetch_text(r["url"])
        if not txt: continue
        contexts.append(txt[:3000])
        sources.append({"title": r["title"], "url": r["url"], "score": None, "kind":"web"})
    ans = _summarize(contexts, query) if contexts else "適切なWeb結果が見つかりませんでした。"
    return {"answer": ans, "sources": sources}


def answer(query: str, mode: str = "doc") -> Dict:
    if mode == "web":
        return _web(query)
    if mode == "hybrid":
        a = _doc(query); b = _web(query)
        contexts = []
        for s in (a["sources"] + b["sources"])[:6]:
            if s.get("kind") == "web" and s.get("url"):
                txt = _fetch_text(s["url"])
                if txt: contexts.append(txt[:1500])
            elif s.get("kind") == "doc" and s.get("path"):
                contexts.append(read_preview(s["path"], limit=1500))
        final = _summarize(contexts, query) if contexts else f"{a['answer']}\n\n{b['answer']}"
        return {"answer": final, "sources": a["sources"] + b["sources"]}
    return _doc(query)
