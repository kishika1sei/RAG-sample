# app/services/vectorstore.py
import os, json
from typing import List, Dict, Tuple
from flask import current_app
import faiss
import numpy as np
from .llm_utils import embed_texts
from .doc_utils import page_count_pdf

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
    pages は以下の優先順位で決める:
      1) meta に total_pages があればそれを優先
      2) page のユニーク数（文字列でも OK）
      3) それでも不明で .pdf かつ実体ありなら pypdf で直接数える（任意）
    """
    if not faiss_exists():
        return []

    _, meta_path = _paths()
    files: Dict[str, Dict] = {}

    def pick_path(m: Dict) -> str:
        for key in ("source", "file", "filepath", "path"):
            v = m.get(key)
            if v:
                return v
        if "doc_id" in m:
            return str(m["doc_id"])
        return "__unknown__"

    def coerce_int(x):
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, float) and x.is_integer():
            return int(x)
        if isinstance(x, str):
            xs = x.strip()
            if xs.isdigit():
                return int(xs)
            import re
            m = re.match(r"^(-?\d+)", xs)
            if m:
                return int(m.group(1))
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            p = pick_path(m)
            if p not in files:
                files[p] = {
                    "path": p,
                    "name": os.path.basename(p) if p != "__unknown__" else "(unknown)",
                    "chunks": 0,
                    "pages_set": set(),
                    "total_pages": m.get("total_pages") or (m.get("metadata") or {}).get("total_pages"),
                }
            files[p]["chunks"] += 1

            # ページ番号（int/str どちらでも拾う）
            pg = m.get("page")
            iv = coerce_int(pg)
            if iv is not None:
                files[p]["pages_set"].add(iv)

            # total_pages が行によって入っているなら拾っておく（未設定時のみ）
            tp = m.get("total_pages") or (m.get("metadata") or {}).get("total_pages")
            if tp is not None and files[p]["total_pages"] is None:
                files[p]["total_pages"] = coerce_int(tp) or tp  # 数値化できれば数値化

    out = []
    for v in files.values():
        pages = v.get("total_pages")
        if pages in (None, 0):
            pages = len(v["pages_set"]) if v["pages_set"] else None

        # 必要なら最終フォールバック（I/Oが重いので任意）
        if pages in (None, 0) and str(v["path"]).lower().endswith(".pdf") and os.path.exists(v["path"]):
            try:
                pages = page_count_pdf(v["path"])
            except Exception:
                pass

        out.append({
            "path": v["path"],
            "name": v["name"],
            "chunks": v["chunks"],
            "pages": pages,
        })

    out.sort(key=lambda x: x["name"])
    return out