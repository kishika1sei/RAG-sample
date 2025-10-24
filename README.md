# セットアップ
pip install -r requirements.txt
cp .env.sample .env  # 値を設定
mkdir -p data/pdf

# 事前インデックス
python scripts/ingest.py  # または UI から /api/ingest を叩く

# 起動
python run.py
# http://localhost:5000
