# app/api.py
from flask import Blueprint, request, jsonify, current_app, g
from werkzeug.utils import secure_filename
from .services.rag import answer
from .services.doc_utils import ingest_local_dir, is_allowed_ext
import os

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.post("/upload")
def api_upload():
    """
    フロントから複数ファイルを受け取り、UPLOAD_DIR(PDF_DIR)へ保存。
    許可拡張子: .pdf .txt .md .markdown
    """
    if "files" not in request.files:
        return jsonify({"ok": False, "error": "files フィールドが見つかりません", "trace_id": getattr(g, "trace_id", "")}), 400

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

    return jsonify({"ok": True, "saved": saved, "skipped": skipped, "upload_dir": upload_dir, "trace_id": getattr(g, "trace_id", "")})


@api_bp.post("/ingest")
def api_ingest():
    """data/pdf を走査してベクトルインデックスを再構築"""
    try:
        n = ingest_local_dir()
        return jsonify({"ok": True, "indexed_docs": n, "trace_id": getattr(g, "trace_id", "")})
    except Exception as e:
        current_app.logger.exception("ingest failed", extra={"trace": {
            "schema_version": 1, "trace_id": getattr(g, "trace_id", ""), "error": str(e), "where": "api_ingest"
        }})
        return jsonify({"ok": False, "error": str(e), "trace_id": getattr(g, "trace_id", "")}), 500


@api_bp.post("/ask")
def api_ask():
    """
    mode=doc|web|hybrid, query=..., debug=bool を受け取りRAGで回答
    debug=true かつ DEBUG_RAG=True の時のみ trace を返す
    """
    data = request.get_json(force=True) if request.is_json else request.form
    query = (data.get("query") or "").strip()
    mode = (data.get("mode") or "doc").lower()
    debug = str(data.get("debug") or "").lower() in ("1", "true", "yes")

    if not query:
        return jsonify({"ok": False, "error": "queryが空です", "trace_id": getattr(g, "trace_id", "")}), 400
    if mode not in ("doc", "web", "hybrid"):
        return jsonify({"ok": False, "error": "modeは doc|web|hybrid のいずれかです", "trace_id": getattr(g, "trace_id", "")}), 400

    try:
        res = answer(query=query, mode=mode, debug=debug)
        return jsonify({"ok": True, **res, "mode": mode, "trace_id": getattr(g, "trace_id", "")})
    except Exception as e:
        current_app.logger.exception("ask failed", extra={"trace": {
            "schema_version": 1, "trace_id": getattr(g, "trace_id", ""), "error": str(e), "where": "api_ask"
        }})
        return jsonify({"ok": False, "error": str(e), "trace_id": getattr(g, "trace_id", "")}), 500
