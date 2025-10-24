# app/routes.py
from flask import Blueprint, render_template, current_app
from .services.vectorstore import list_indexed_files, faiss_exists

web_bp = Blueprint("web", __name__)

@web_bp.get("/")
def index():
    files = []
    has_index = faiss_exists()
    try:
        if has_index:
            files = list_indexed_files()
    except Exception:
        current_app.logger.exception("failed to list indexed files")
    return render_template("index.html", files=files, has_index=has_index)
