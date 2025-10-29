# app/services/llm_utils.py
import os
import time
from typing import List, Dict, Tuple, Any, Optional
from openai import OpenAI

_client_singleton: Optional[OpenAI] = None

def get_client() -> OpenAI:
    global _client_singleton
    if _client_singleton is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY が未設定です（.env を確認）")
        _client_singleton = OpenAI(api_key=key)  # envから拾う場合は OpenAI() でもOK
    return _client_singleton

# ====== 埋め込み ======

def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    """後方互換：埋め込みのみ返す（計測・usageは不要な場面向け）"""
    embs, _meta = embed_texts_with_meta(texts, model)
    return embs

def embed_texts_with_meta(texts: List[str], model: str) -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    計測＆usage付き。戻り値:
      (embeddings, {"ms": int, "usage": {...} or None, "model": str})
    """
    cli = get_client()
    t0 = time.perf_counter()
    resp = cli.embeddings.create(model=model, input=texts)
    ms = int((time.perf_counter() - t0) * 1000)

    embs = [d.embedding for d in resp.data]
    usage = getattr(resp, "usage", None)
    if usage is not None:
        # openai-python v1系は pydantic objects → dict() で扱いやすく
        try:
            usage = usage.model_dump()  # pydantic v2
        except Exception:
            usage = dict(usage)

    meta = {"ms": ms, "usage": usage, "model": model}
    return embs, meta

# ====== チャット補完 ======

def chat(messages: List[Dict], model: str) -> str:
    """後方互換：本文のみ返す"""
    text, _meta = chat_with_meta(messages=messages, model=model)
    return text

def chat_with_meta(messages: List[Dict], model: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
    """
    計測＆usage付きチャット。戻り値:
      (text, {"ms": int, "usage": {...} or None, "finish_reason": str|None, "id": str|None, "model": str})
    """
    cli = get_client()
    t0 = time.perf_counter()
    resp = cli.chat.completions.create(
        model=model,
        messages=messages,
        temperature=kwargs.pop("temperature", 0.2),
        **kwargs
    )
    ms = int((time.perf_counter() - t0) * 1000)

    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    finish_reason = getattr(choice, "finish_reason", None)
    resp_id = getattr(resp, "id", None)

    usage = getattr(resp, "usage", None)
    if usage is not None:
        try:
            usage = usage.model_dump()
        except Exception:
            usage = dict(usage)

    meta = {
        "ms": ms,
        "usage": usage,
        "finish_reason": finish_reason,
        "id": resp_id,
        "model": model,
    }
    return text, meta
