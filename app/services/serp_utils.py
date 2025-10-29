# serp_utils.py
# 依存: pip install serpapi python-dotenv
import os
import time
from typing import Dict, Any, List, Optional, Iterable, TypedDict
from urllib.parse import urlparse

from dotenv import load_dotenv
from serpapi import GoogleSearch

load_dotenv()

# === 環境変数からAPIキー取得（両表記に対応） ===
def get_serpapi_key() -> str:
    key = os.getenv("SERP_API_KEY") or os.getenv("SERPAPI_API_KEY")
    if not key:
        raise RuntimeError(
            "SerpAPIキーが未設定です。'.env' に SERP_API_KEY=... "
            "（または SERPAPI_API_KEY=...）を追加してください。"
        )
    return key

# === 正規化済み検索結果の型 ===
class Hit(TypedDict, total=False):
    position: int
    title: str
    url: str
    snippet: str
    source: str  # ニュースなどで使う
    date: str    # ニュースなどで使う

# === 指数バックオフ付きの呼び出し ===
def serpapi_call(params: Dict[str, Any], retries: int = 4, timeout: float = 10.0) -> Dict[str, Any]:
    backoff = 1.0
    last_err: Optional[Exception] = None
    for _ in range(retries):
        try:
            search = GoogleSearch(params)
            # GoogleSearch側でtimeoutは管理されるため、明示的timeoutは不要
            return search.get_dict()
        except Exception as e:
            last_err = e
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(f"SerpAPI呼び出しに失敗: {repr(last_err)}")

# === URLの簡易正規化（重複除去用） ===
def url_key(u: str) -> str:
    p = urlparse(u or "")
    return f"{p.netloc}{p.path}"

# === ドメインの多様性を確保（同一ドメイン連続を間引く簡易版） ===
def diversify_by_domain(hits: List[Hit], max_per_domain: int = 2) -> List[Hit]:
    seen: Dict[str, int] = {}
    out: List[Hit] = []
    for h in hits:
        dom = urlparse(h.get("url", "")).netloc
        if not dom:
            out.append(h)
            continue
        count = seen.get(dom, 0)
        if count < max_per_domain:
            out.append(h)
            seen[dom] = count + 1
    return out

# === organic_results の正規化 ===
def normalize_organic(data: Dict[str, Any]) -> List[Hit]:
    out: List[Hit] = []
    for item in (data.get("organic_results") or []):
        out.append(Hit(
            position=item.get("position"),
            title=item.get("title"),
            url=item.get("link"),
            snippet=item.get("snippet"),
        ))
    return out

# === news_results の正規化 ===
def normalize_news(data: Dict[str, Any]) -> List[Hit]:
    out: List[Hit] = []
    for i, item in enumerate((data.get("news_results") or []), start=1):
        out.append(Hit(
            position=i,
            title=item.get("title"),
            url=item.get("link") or item.get("news_url"),
            snippet=item.get("snippet"),
            source=item.get("source"),
            date=item.get("date"),
        ))
    return out

# === Google検索（ページング対応・重複除去・ドメイン多様性） ===
def google_search(q: str, *, hl: str = "ja", gl: str = "jp", safe: str = "active",
                  num: int = 10, pages: int = 1, tbs: Optional[str] = None) -> List[Hit]:
    api_key = get_serpapi_key()
    all_hits: List[Hit] = []
    seen_urls = set()

    for p in range(pages):
        params = {
            "engine": "google",
            "q": q,
            "api_key": api_key,
            "hl": hl,
            "gl": gl,
            "safe": safe,
            "num": num,
            "start": p * num,
        }
        if tbs:
            params["tbs"] = tbs  # 期間指定など

        data = serpapi_call(params)
        hits = normalize_organic(data)

        # URL重複を弾く
        for h in hits:
            k = url_key(h.get("url", ""))
            if k and k in seen_urls:
                continue
            seen_urls.add(k)
            all_hits.append(h)

        # 件数が埋まらなければ打ち切り（次ページなし）
        if len(hits) < num:
            break

    # ドメイン多様性をざっくり確保
    return diversify_by_domain(all_hits, max_per_domain=2)

# === Googleニュース検索 ===
def google_news(q: str, *, hl: str = "ja", gl: str = "jp", num: int = 10) -> List[Hit]:
    api_key = get_serpapi_key()
    params = {
        "engine": "google_news",
        "q": q,
        "api_key": api_key,
        "hl": hl,
        "gl": gl,
        "num": num
    }
    data = serpapi_call(params)
    return normalize_news(data)

# === クエリテンプレ ===
def q_site(keyword: str, site: str) -> str:
    return f'site:{site} "{keyword}"'

def q_pdf(keyword: str) -> str:
    return f'"{keyword}" filetype:pdf'

def q_recent(keyword: str, unit: str = "w") -> Dict[str, Any]:
    # unit: d（日）/ w（週）/ m（月）/ y（年）
    return {"q": f'"{keyword}"', "tbs": f"qdr:{unit}"}
