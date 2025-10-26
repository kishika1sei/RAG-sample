# app/services/rag.py
from typing import Dict, List, Any
from flask import current_app, g
from .llm_utils import chat, chat_with_meta
from .vectorstore import faiss_exists, faiss_search
from .doc_utils import read_preview
from .serp_utils import google_search
import requests
from bs4 import BeautifulSoup
import time

# 初期プロンプト（設定で差し替え可）
DEFAULT_SYS = "あなたは日本語で正確に答えるアシスタントです。根拠に基づき簡潔に回答し、不明な点は正直に『不明』と述べてください。"


def _fetch_text(url: str, timeout: int = 10) -> str:
    try:
        html = requests.get(url, timeout=timeout).text
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n")
    except Exception:
        return ""


# def _summarize(contexts: List[str], query: str, timing: Dict[str, int] = None, steps: Dict[str, Any] = None) -> str:
#     sys_msg = current_app.config.get("SYS_PROMPT", DEFAULT_SYS)
#     user = (
#         "以下のコンテキストを根拠に質問へ回答してください。"
#         "不足していれば『不明』と記してください。\n\n"
#         f"【質問】\n{query}\n\n【コンテキスト】\n" + "\n---\n".join(contexts[:8])
#     )
#     messages = [  # ★毎回新規作成（履歴を持ち回らない）
#         {"role": "system", "content": sys_msg},
#         {"role": "user", "content": user},
#     ]
#     text, meta = chat_with_meta(
#         messages = messages,
#         # =[{"role":"system","content":sys_msg}, {"role":"user","content":user}],
#         model=current_app.config["LLM_MODEL"]
#     )
#     if timing is not None:
#         timing["llm_ms"] = timing.get("llm_ms", 0) + int(meta.get("ms", 0))
#     if steps is not None and meta.get("usage"):
#         steps["usage"] = meta["usage"]
#     return text

def _summarize(contexts: List[str], query: str,
               timing: Dict[str, int] = None, steps: Dict[str, Any] = None) -> str:
    sys_msg = current_app.config.get("SYS_PROMPT", DEFAULT_SYS)
    llm_model = current_app.config["LLM_MODEL"]
    max_chunks = current_app.config.get("CTX_MAX_CHUNKS", 8)
    max_chars_per_chunk = current_app.config.get("CTX_MAX_CHARS", 3000)

    # 軽い正規化＆上限
    normed = []
    for c in contexts[:max_chunks]:
        if not c: continue
        s = " ".join(c.split())[:max_chars_per_chunk]  # 連続空白圧縮 + 文字上限
        if s: normed.append(s)

    user = (
        "以下のコンテキストを根拠に質問へ回答してください。"
        "不足していれば『不明』と記してください。\n\n"
        f"【質問】\n{query}\n\n【コンテキスト】\n" + "\n---\n".join(normed)
    )
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user},
    ]

    text, meta = chat_with_meta(messages=messages, model=llm_model)

    if timing is not None:
        timing["llm_ms"] = timing.get("llm_ms", 0) + int(meta.get("ms", 0))
    if steps is not None:
        if meta.get("usage"):
            steps["usage"] = meta["usage"]
        steps["llm_model_used"] = llm_model  # 任意だが便利

    return text



def _context_preview_from_doc_hits(doc_hits: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    out = []
    for h in (doc_hits or [])[:limit]:
        name = h.get("doc") or h.get("name") or h.get("file") or "document"
        page = h.get("page", "-")
        snippet = (read_preview(h.get("path", ""), limit=300) or h.get("text") or h.get("snippet") or "")[:200]
        out.append(f"[{name} p.{page}] {snippet}")
    return out


def _context_preview_from_web_hits(web_hits: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    out = []
    for i, r in enumerate((web_hits or [])[:limit], start=1):
        snippet = (r.get("snippet") or "")[:200]
        title = r.get("title") or r.get("url") or "web"
        out.append(f"[web {i}] {title} — {snippet}")
    return out

def _dedup_preview(items: List[str], limit: int = 3) -> List[str]:
    seen, out = set(), []
    for s in items or []:
        key = s.split("]")[0]  # "[foo p.3" までで重複判定
        if key in seen:
            continue
        seen.add(key)
        out.append(" ".join(s.split()))  # 連続空白/改行の簡易正規化
    return out[:limit]


def _summarize_sources(doc_hits: List[Dict[str, Any]], web_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources = []
    for h in doc_hits or []:
        sources.append({
            "title": h.get("doc") or h.get("name") or h.get("file") or "document",
            "kind": "doc",
            "score": h.get("score"),
            "page": h.get("page"),
            "path": h.get("path")
        })
    for r in web_hits or []:
        sources.append({
            "title": r.get("title") or r.get("url"),
            "kind": "web",
            "score": r.get("score"),
            "url": r.get("url")
        })
    return sources


def _doc(query: str, params: Dict[str, Any], timing: Dict[str, int], steps: Dict[str, Any]) -> Dict[str, Any]:
    if not faiss_exists():
        msg = "インデックスがありません。先に /api/ingest を実行してください。"
        steps["doc_hits"] = []
        return {"answer": msg, "doc_hits": [], "sources": []}

    t = time.perf_counter()
    hits = faiss_search(query, k=params["top_k"])
    timing["retrieval_ms_doc"] = int((time.perf_counter() - t) * 1000)

    # ★ ここでパスを正規化（\ → /）
    for h in hits:
        p = h.get("path")
        if isinstance(p, str):
            h["path"] = p.replace("\\", "/")

    steps["doc_hits"] = hits
    # コンテキスト作成（上位3つを採用）
    contexts, sources = [], []
    for h in hits[:3]:
        preview = read_preview(h.get("path", ""), limit=3000)
        if preview:
            contexts.append(preview)
        sources.append({
            "title": h.get("doc") or h.get("name") or h.get("file") or "document",
            "url": None,
            "score": h.get("score"),
            "kind": "doc",
            "path": h.get("path"),  # ← 既に正規化済み
            "page": h.get("page")
        })

    ans = _summarize(contexts, query, timing=timing, steps=steps) if contexts else "該当ドキュメントが見つかりませんでした。"
    steps["context_preview_doc"] = _context_preview_from_doc_hits(hits)
    return {"answer": ans, "doc_hits": hits, "sources": sources}



def _web(query: str, params: Dict[str, Any], timing: Dict[str, int], steps: Dict[str, Any]) -> Dict[str, Any]:
    t = time.perf_counter()
    results = google_search(query, pages=1)  # 返却: [{'title','url',...}] を想定
    timing["retrieval_ms_web"] = int((time.perf_counter() - t) * 1000)

    contexts, sources, web_hits = [], [], []
    for rank, r in enumerate(results[:params["top_k"]], start=1):
        url = r.get("url")
        txt = _fetch_text(url) if url else ""
        if not txt:
            continue
        snippet = r.get("snippet") or (txt[:240] if txt else "")
        contexts.append(txt[:3000])
        h = {"title": r.get("title"), "url": url, "rank": rank, "score": r.get("score"), "snippet": snippet}
        web_hits.append(h)
        sources.append({"title": r.get("title"), "url": url, "score": r.get("score"), "kind": "web"})

    steps["web_hits"] = web_hits
    steps["context_preview_web"] = _context_preview_from_web_hits(web_hits)
    ans = _summarize(contexts, query, timing=timing, steps=steps) if contexts else "適切なWeb結果が見つかりませんでした。"
    return {"answer": ans, "web_hits": web_hits, "sources": sources}


def answer(query: str, mode: str = "doc", debug: bool = False) -> Dict[str, Any]:
    """
    主要なトレースを構造化して返す（debug=True かつ DEBUG_RAG=True のときのみ trace 同梱）
    """
    t0 = time.perf_counter()
    params: Dict[str, Any] = {
        "mode": mode,
        "top_k": current_app.config.get("RAG_TOP_K", 5),
        "threshold": current_app.config.get("RAG_THRESHOLD", 0.0),
        "embed_model": current_app.config.get("EMBED_MODEL"),
        "llm_model": current_app.config.get("LLM_MODEL"),
    }
    timing: Dict[str, int] = {}
    steps: Dict[str, Any] = {"query": query}

    # ルートごとの実処理
    if mode == "web":
        w = _web(query, params, timing, steps)
        answer_text = w["answer"]
        doc_hits, web_hits = [], w.get("web_hits", [])
        sources = w.get("sources", [])
        failover = None
    elif mode == "hybrid":
        d = _doc(query, params, timing, steps)
        w = _web(query, params, timing, steps)
        doc_hits, web_hits = d.get("doc_hits", []), w.get("web_hits", [])
        # 再要約（doc + web）: それぞれの出典から軽量文脈を再構築
        contexts: List[str] = []
        for s in (d.get("sources", []) + w.get("sources", []))[:6]:
            if s.get("kind") == "web" and s.get("url"):
                txt = _fetch_text(s["url"])
                if txt:
                    contexts.append(txt[:1500])
            elif s.get("kind") == "doc" and s.get("path"):
                prev = read_preview(s["path"], limit=1500)
                if prev:
                    contexts.append(prev)
        if contexts:
            answer_text = _summarize(contexts, query, timing=timing, steps=steps)
        else:
            answer_text = f"{d['answer']}\n\n{w['answer']}"
        sources = d.get("sources", []) + w.get("sources", [])
        failover = ("doc→web" if (not doc_hits and web_hits) else
                    "web→doc" if (not web_hits and doc_hits) else None)
    else:
        d = _doc(query, params, timing, steps)
        answer_text = d["answer"]
        doc_hits, web_hits = d.get("doc_hits", []), []
        sources = d.get("sources", [])
        failover = None

    # LLM 使用量は llm_utils.chat 側が返せるなら steps["usage"] に格納済みが理想
    # ここでは無い前提でスキップ（実装できるなら入れてOK）

    timing["total_ms"] = int((time.perf_counter() - t0) * 1000)
    steps["failover"] = failover

    # UI用出典の要約（score/url/page などをそろえる）
    ui_sources = _summarize_sources(doc_hits, web_hits)

    # 合成 → 重複除去＆整形(Webとdocのコンテキストを結合)
    _contexts_combined = (steps.get("context_preview_doc", []) +
                        steps.get("context_preview_web", []))
    _contexts_combined = _dedup_preview(_contexts_combined, limit=6)

    # trace を構築
    trace = {
        "schema_version": 1,
        "trace_id": getattr(g, "trace_id", ""),
        "params": params,
        "timing": timing,
        "steps": {
            "query": steps.get("query"),
            "doc_hits": steps.get("doc_hits", []),
            "web_hits": steps.get("web_hits", []),
            "context_preview": _contexts_combined,
            # promptはデフォルト非表示（安全策）
            "prompt": "[hidden]",
            "usage": steps.get("usage"),
            "failover": steps.get("failover")
        }
    }

    # ログへ（NDJSON）。prompt は既定で出さない
    current_app.logger.info("rag.trace", extra={"trace": trace})

    # レスポンス用 payload
    payload: Dict[str, Any] = {"answer": answer_text, "sources": ui_sources}

    # debug=true かつ DEBUG_RAG=True のときだけ trace を返す
    if debug and current_app.config.get("DEBUG_RAG", False):
        # 必要なら prompt を開示（ただし機微混入に注意）
        # trace["steps"]["prompt"] = 実際のプロンプト文字列
        payload["trace"] = trace

    return payload
