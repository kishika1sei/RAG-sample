# config.py
import os

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    PDF_DIR = os.getenv("PDF_DIR", "data/pdf")
    INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
    CTX_MAX_CHUNKS = 4
    CTX_MAX_CHARS = 1500
    SYS_PROMPT = os.getenv("SYS_PROMPT", "あなたは日本語で正確に答えるアシスタントです。根拠に基づき簡潔に回答し、不明な点は正直に『不明』と述べてください。")