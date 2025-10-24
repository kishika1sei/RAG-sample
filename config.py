# config.py
import os

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    PDF_DIR = os.getenv("PDF_DIR", "data/pdf")
    INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
