# app/services/llm_utils.py
import os
from typing import List, Dict
from openai import OpenAI

_client_singleton: OpenAI | None = None

# クライアント情報を取得
# APIキーを環境変数から読み込む
def get_client() -> OpenAI:
    global _client_singleton
    if _client_singleton is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY が未設定です（.env を確認）")
        _client_singleton = OpenAI(api_key=key)  # or OpenAI() でもOK（環境変数使用）
    return _client_singleton

# テキスト群を埋め込みベクトルに変換（ベクトルDB格納や検索用）
def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    cli = get_client()
    resp = cli.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

# チャット補完APIを呼び出し、応答テキスト（content）を返す
def chat(messages: List[Dict], model: str) -> str:
    cli = get_client()
    resp = cli.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return resp.choices[0].message.content.strip()
