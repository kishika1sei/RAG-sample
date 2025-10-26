# app/__init__.py
from flask import Flask, g, request
from .routes import web_bp
from .api import api_bp  # .以降の部分はpythonのファイル名が入る
from config import Config  # ← ルート直下の config.py を参照
import logging, json ,sys , uuid ,time

class JsonFormatter(logging.Formatter):
    def format(self, record):
        base = {"level": record.levelname, "msg": record.getMessage()}
        # extra={"trace": {...}} をそのまま展開
        if hasattr(record, "trace"):
            base.update(record.trace)
        return json.dumps(base, ensure_ascii=False)
    
# Flaskアプリを作る
def create_app():
    # Flask本体を生成し、config.pyを読み込む
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(Config)

    # 構造化ログ
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    app.logger.handlers[:] = [h]
    app.logger.setLevel(logging.INFO)

    # リクエスト共通のtrace_id と 計測開始
    @app.before_request
    def _before():
        g.trace_id = uuid.uuid4().hex[:12]
        g.t0 = time.perf_counter()

    @app.after_request
    def _after(resp):
        # レイテンシを軽く載せる（詳細はrag.pyで）
        total_ms = int((time.perf_counter()-g.get("t0", time.perf_counter()))*1000)
        resp.headers["X-Trace-Id"] = g.get("trace_id","")
        resp.headers["X-RTT-Ms"] = str(total_ms)
        return resp

    @app.errorhandler(Exception)
    def _err(e):
        app.logger.exception("unhandled", extra={"trace": {
            "schema_version": 1,
            "trace_id": getattr(g, "trace_id", ""),
            "error": str(e),
            "path": request.path,
            "method": request.method,
        }})
        return {"ok": False, "error": "internal error", "trace_id": getattr(g, "trace_id","")}, 500


    #BluePrint登録
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    return app
