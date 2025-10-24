# app/services/vectorstore.py
import os, json
from typing import List, Dict, Tuple
from flask import current_app
import faiss
import numpy as np
from .llm_utils import embed_texts

# インデックス保存先ディレクトリを作成し、
# FAISSバイナリ(index)とメタデータ(JSON Lines)の各パスを返す
def _paths() -> Tuple[str, str]:
    idx_dir = current_app.config["INDEX_DIR"]
    os.makedirs(idx_dir, exist_ok=True)
    return os.path.join(idx_dir, "faiss.index"), os.path.join(idx_dir, "meta.jsonl")

# FAISSのインデックスファイルとメタデータ(JSONL)が両方存在するかを確認
def faiss_exists() -> bool:
    idx, meta = _paths()
    return os.path.exists(idx) and os.path.exists(meta)


# ベクトル群と対応メタデータを受け取り、FAISSインデックス(内積)＋JSONLメタを保存する
def faiss_save(vectors: List[List[float]], metas: List[Dict]):
    import numpy as np, faiss, os, json
    dim = len(vectors[0]) if vectors else 0
    arr = np.array(vectors, dtype="float32")
    # ★ 正規化（L2ノルム1に）
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    arr = arr / norms

    index = faiss.IndexFlatIP(dim)
    index.add(arr)
    idx_path, meta_path = _paths()
    faiss.write_index(index, idx_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


# クエリを埋め込み→L2正規化→内積(IndexFlatIP)で上位k件を検索し、scoreとメタを返す
def faiss_search(query: str, k: int = 5) -> List[Dict]:
    """クエリを埋め込み→内積で上位k件返却"""
    idx_path, meta_path = _paths()
    index = faiss.read_index(idx_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metas = [json.loads(l) for l in f]

    qv = embed_texts([query], model=current_app.config["EMBED_MODEL"])[0]
    qv = np.asarray(qv, dtype="float32")
    qv = qv / (np.linalg.norm(qv) + 1e-12) 
    D, I = index.search(np.array([qv], dtype="float32"), k)
    out = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1: continue
        m = metas[idx]
        out.append({"score": float(score), **m})
    return out

def list_indexed_files() -> List[Dict]:
    """
    meta.jsonl を走査して、取り込まれているファイルの一覧を返す。
    返り値例:
    [
      {"path": "/abs/path/to/data/pdf/foo.pdf", "name": "foo.pdf", "chunks": 128, "pages": 10},
      {"path": "/abs/path/to/data/pdf/bar.pdf", "name": "bar.pdf", "chunks": 64, "pages": 5},
    ]
    pages はメタに page があれば集計、なければ None
    """
    if not faiss_exists():
        return []

    _, meta_path = _paths()
    files: Dict[str, Dict] = {}

    def pick_path(m: Dict) -> str:
        # メタにありそうなキーを順に当てる（実装差吸収）
        for key in ("source", "file", "filepath", "path"):
            if key in m and m[key]:
                return m[key]
        # どうしても無ければ doc_id でグルーピング（ファイル名は doc_id を仮名）
        if "doc_id" in m:
            return str(m["doc_id"])
        return "__unknown__"

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            p = pick_path(m)
            if p not in files:
                files[p] = {
                    "path": p,
                    "name": os.path.basename(p) if p != "__unknown__" else "(unknown)",
                    "chunks": 0,
                    "pages_set": set(),  # 後で len に変換
                }
            files[p]["chunks"] += 1
            # page が取れるなら集計（無ければスキップ）
            pg = m.get("page")
            if isinstance(pg, int):
                files[p]["pages_set"].add(pg)

    # 整形
    out = []
    for v in files.values():
        pages = len(v["pages_set"]) if v["pages_set"] else None
        out.append({
            "path": v["path"],
            "name": v["name"],
            "chunks": v["chunks"],
            "pages": pages,
        })

    # 表示は名前順に
    out.sort(key=lambda x: x["name"])
    return out