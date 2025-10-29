# app/api.py
from flask import Blueprint, request, jsonify, current_app, g
from werkzeug.utils import secure_filename as wz_secure_filename  # 既存のままでもOK
from .services.rag import answer
from .services.doc_utils import ingest_local_dir, is_allowed_ext
import os
import unicodedata
import re
import secrets

api_bp = Blueprint("api", __name__, url_prefix="/api")

def safe_filename_keep_unicode(filename: str) -> str:
    """
    日本語などのUnicodeを残しつつ、安全なファイル名に整える。
    - ディレクトリ区切りや制御文字を除去
    - 正規化(NFKC)
    - 長すぎるファイル名を適度にカット
    """
    # ベース名だけに（Windows由来のパス混入対策）
    name = os.path.basename(filename)

    # 正規化（全角→半角などの揺れを吸収）
    name = unicodedata.normalize("NFKC", name)

    # 制御文字の除去
    name = re.sub(r'[\x00-\x1f\x7f]+', '', name)

    # パス区切り記号の無効化
    name = name.replace('/', '_').replace('\\', '_')

    # 先頭や末尾のドット/空白を除去（隠しファイル化やトリム事故回避）
    name = name.strip().strip('.')

    # 無名対策
    if not name:
        name = "file"

    # 長すぎる場合はベース名を縮める
    base, ext = os.path.splitext(name)
    MAX_BASE = 150  # OS制限をざっくり考慮
    if len(base) > MAX_BASE:
        base = base[:MAX_BASE]
    # 拡張子はそのまま（is_allowed_ext で判定するため）
    return base + ext

def uniquify_path(dirpath: str, filename: str) -> str:
    """重複があれば _1, _2 ... と連番を付けて衝突回避"""
    base, ext = os.path.splitext(filename)
    candidate = filename
    i = 1
    while os.path.exists(os.path.join(dirpath, candidate)):
        candidate = f"{base}_{i}{ext}"
        i += 1
    return os.path.join(dirpath, candidate)

@api_bp.post("/upload")
def api_upload():
    if "files" not in request.files:
        return jsonify({"ok": False, "error": "files フィールドが見つかりません", "trace_id": getattr(g, "trace_id", "")}), 400

    upload_dir = current_app.config.get("PDF_DIR", "data/pdf")
    os.makedirs(upload_dir, exist_ok=True)

    files = request.files.getlist("files")
    saved, skipped = [], []
    for f in files:
        if not f.filename:
            continue

        # ← ここを secure_filename から置き換え
        filename_raw = f.filename
        filename = safe_filename_keep_unicode(filename_raw)

        # 許可拡張子チェック（拡張子は normalize 後のものを使用）
        if not is_allowed_ext(filename):
            skipped.append({"name": filename_raw, "reason": "拡張子が許可されていません"})
            continue

        # 同名衝突を回避
        path = uniquify_path(upload_dir, filename)

        # 保存
        f.save(path)
        saved.append(os.path.basename(path))

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

@api_bp.post("/reset")
def api_reset():
    """
    （任意）サーバ側に会話スコープがある場合のみ。
    セッションやDB上の会話履歴・一時状態を初期化。
    """
    try:
        # 例: session["conv"] = {"id": new_id, "messages": [], ...}
        # 今の実装に会話スコープが無いならダミーでOK
        return jsonify({"ok": True, "message": "reset done", "trace_id": getattr(g, "trace_id", "")})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace_id": getattr(g, "trace_id", "")}), 500