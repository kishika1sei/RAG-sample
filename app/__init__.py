# app/__init__.py
from flask import Flask
from .routes import web_bp
from .api import api_bp  # .以降の部分はpythonのファイル名が入る
from config import Config  # ← ルート直下の config.py を参照

# Flaskアプリを作る
def create_app():
    # Flask本体を生成し、config.pyを読み込む
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(Config)

    #BluePrint登録
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    return app
