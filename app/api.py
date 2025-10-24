# app/api.py
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from .services.rag import answer
from .services.doc_utils import ingest_local_dir,is_allowed_ext
import os
api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.post("/upload")
def api_upload():
    """
    フロントから複数ファイルを受け取り、UPLOAD_DIR(PDF_DIR)へ保存。
    許可拡張子: .pdf .txt .md .markdown
    """
    if "files" not in request.files:
        return jsonify({"ok": False, "error": "files フィールドが見つかりません"}), 400

    upload_dir = current_app.config.get("PDF_DIR", "data/pdf")
    os.makedirs(upload_dir, exist_ok=True)

    files = request.files.getlist("files")
    saved, skipped = [], []
    for f in files:
        if not f.filename:
            continue
        filename = secure_filename(f.filename)
        if not is_allowed_ext(filename):
            skipped.append({"name": filename, "reason": "拡張子が許可されていません"})
            continue
        path = os.path.join(upload_dir, filename)
        f.save(path)
        saved.append(filename)

    return jsonify({"ok": True, "saved": saved, "skipped": skipped, "upload_dir": upload_dir})

@api_bp.post("/ingest")
def api_ingest():
    """data/pdf を走査してベクトルインデックスを再構築"""
    try:
        n = ingest_local_dir()
        return jsonify({"ok": True, "indexed_docs": n})
    except Exception as e:
        current_app.logger.exception("ingest failed")
        return jsonify({"ok": False, "error": str(e)}), 500

@api_bp.post("/ask")
def api_ask():
    """mode=doc|web|hybrid, query=... を受け取りRAGで回答"""
    data = request.get_json(force=True) if request.is_json else request.form
    query = (data.get("query") or "").strip()
    mode = (data.get("mode") or "doc").lower()
    if not query:
        return jsonify({"ok": False, "error": "queryが空です"}), 400
    if mode not in ("doc", "web", "hybrid"):
        return jsonify({"ok": False, "error": "modeは doc|web|hybrid のいずれかです"}), 400
    try:
        res = answer(query=query, mode=mode)
        return jsonify({"ok": True, **res, "mode": mode})
    except Exception as e:
        current_app.logger.exception("ask failed")
        return jsonify({"ok": False, "error": str(e)}), 500
