"""Ba phương pháp retrieval cho Day 7: dense (pgvector Day 6), BM25 (lexical), hybrid RRF.

Mọi retriever có cùng giao diện:
    search(query: str, top_k: int = 3, filter: dict | None = None) -> list[dict]
và trả về list chunk theo schema Day 7, kèm `score`, `rank` (1-based) và `method`:
    {id, text, source_id, source, page, metadata, score, rank, method, ...}

`filter` là thông tin người dùng cung cấp sẵn trong câu hỏi (ví dụ {"chuong": "VIII"}),
KHÔNG được suy ra từ gold. Retriever không bao giờ nhận gold hay expected_answer.
"""
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILTER_KEYS = ("chuong", "muc", "dieu")  # cột metadata có thật trong bảng chunks của Day 6


def norm(s: str) -> str:
    """NFC + gộp khoảng trắng; dùng để so khớp evidence (PDF hay ngắt dòng giữa câu)."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "")).strip()


def tokenize(text: str) -> list[str]:
    """Tách từ đơn giản cho tiếng Việt: NFC, chữ thường, tách theo ký tự chữ/số (âm tiết)."""
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def load_chunks(path: Path = ROOT / "data" / "chunks.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check_filter(filter: dict | None) -> dict:
    filter = dict(filter or {})
    bad = set(filter) - set(FILTER_KEYS)
    if bad:
        raise ValueError(f"Filter không hỗ trợ: {sorted(bad)}; chỉ dùng {FILTER_KEYS}")
    return filter


def _match(chunk: dict, filter: dict) -> bool:
    return all(chunk["metadata"].get(k) == v for k, v in filter.items())


def _result(chunk: dict, score: float, rank: int, method: str, **extra) -> dict:
    return {**chunk, "score": float(score), "rank": rank, "method": method, **extra}


class DenseRetriever:
    """Dense retrieval trên vector store Day 6 (PostgreSQL + pgvector, cosine, exact search).

    Query được embed bằng đúng model đã dùng để index ở Day 6 (EMBEDDING_MODEL trong .env).
    Không có fallback: lỗi DB/model sẽ được raise để ghi vào trace, không thay bằng kết quả giả.
    """

    name = "dense"

    def __init__(self, chunks: list[dict], db_url: str | None = None,
                 model_name: str | None = None, strategy: str | None = None):
        import psycopg
        from dotenv import load_dotenv
        from sentence_transformers import SentenceTransformer

        load_dotenv(ROOT / ".env")
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL")
        self.strategy = strategy or os.getenv("PG_STRATEGY", "B")
        if not self.db_url or not self.model_name:
            raise RuntimeError("Thiếu DATABASE_URL hoặc EMBEDDING_MODEL trong .env")

        self.by_id = {c["id"]: c for c in chunks}
        self.model = SentenceTransformer(self.model_name, device="cpu")
        self.conn = psycopg.connect(self.db_url, connect_timeout=5)
        stored_dim = self.conn.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'").fetchone()[0]
        dim = int(self.model.get_sentence_embedding_dimension())
        if dim != stored_dim:
            raise RuntimeError(f"Model cho {dim} chiều nhưng index lưu vector({stored_dim})")
        self.dim = dim

    def search(self, query: str, top_k: int = 3, filter: dict | None = None) -> list[dict]:
        filter = _check_filter(filter)
        qv = self.model.encode([query], normalize_embeddings=True)[0]
        vec = "[" + ",".join(f"{x:.7f}" for x in qv) + "]"
        where, params = "strategy = %s", [self.strategy]
        for k, v in filter.items():  # tên cột lấy từ whitelist FILTER_KEYS, giá trị truyền qua tham số
            where += f" AND {k} = %s"
            params.append(v)
        rows = self.conn.execute(
            f"SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score FROM chunks "
            f"WHERE {where} ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s",
            [vec, *params, vec, top_k]).fetchall()
        out = []
        for rank, (chunk_id, score) in enumerate(rows, 1):
            if chunk_id not in self.by_id:
                raise RuntimeError(f"Chunk {chunk_id} có trong DB nhưng không có trong chunks.json; "
                                   "chạy lại src.day6_export")
            out.append(_result(self.by_id[chunk_id], score, rank, self.name))
        return out

    def close(self):
        self.conn.close()


class BM25Retriever:
    """BM25 (Okapi) trên data/chunks.json, tự cài đặt, không cần thư viện ngoài.

    Tham số mặc định k1=1.5, b=0.75. Tokenizer tách theo âm tiết, không tách từ ghép tiếng Việt.
    Filter được áp trước khi xếp hạng (chỉ chấm điểm chunk thỏa filter).
    """

    name = "bm25"

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tfs = [Counter(tokenize(c["text"])) for c in chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / len(self.lens)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def _score(self, q_terms: list[str], i: int) -> float:
        tf, dl, s = self.tfs[i], self.lens[i], 0.0
        for t in q_terms:
            f = tf.get(t)
            if f:
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query: str, top_k: int = 3, filter: dict | None = None) -> list[dict]:
        filter = _check_filter(filter)
        q_terms = tokenize(query)
        scored = [(self._score(q_terms, i), c) for i, c in enumerate(self.chunks)
                  if _match(c, filter)]
        scored = [x for x in scored if x[0] > 0]
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        return [_result(c, s, r, self.name) for r, (s, c) in enumerate(scored[:top_k], 1)]


class HybridRRFRetriever:
    """Hybrid: hợp nhất ranking của dense và BM25 bằng Reciprocal Rank Fusion.

    score(d) = sum_{retriever} 1 / (rrf_k + rank_retriever(d)), mỗi retriever lấy `candidates` chunk.
    Chỉ dùng thứ hạng nên không cần chuẩn hóa score giữa cosine và BM25.
    """

    name = "hybrid_rrf"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever,
                 candidates: int = 20, rrf_k: int = 60):
        self.dense, self.bm25, self.candidates, self.rrf_k = dense, bm25, candidates, rrf_k

    def search(self, query: str, top_k: int = 3, filter: dict | None = None) -> list[dict]:
        n = max(self.candidates, top_k)
        lists = {"dense": self.dense.search(query, n, filter),
                 "bm25": self.bm25.search(query, n, filter)}
        fused: dict[str, dict] = {}
        for src, results in lists.items():
            for r in results:
                item = fused.setdefault(r["id"], {"chunk": r, "score": 0.0, "ranks": {}})
                item["score"] += 1 / (self.rrf_k + r["rank"])
                item["ranks"][src] = r["rank"]
        ordered = sorted(fused.values(), key=lambda x: (-x["score"], x["chunk"]["id"]))[:top_k]
        out = []
        for rank, item in enumerate(ordered, 1):
            base = {k: v for k, v in item["chunk"].items() if k not in ("score", "rank", "method")}
            out.append(_result(base, item["score"], rank, self.name,
                               dense_rank=item["ranks"].get("dense"),
                               bm25_rank=item["ranks"].get("bm25")))
        return out


def build_retrievers(methods: list[str], chunks: list[dict] | None = None) -> dict:
    """Tạo các retriever theo tên; dense chỉ khởi tạo (kết nối DB + load model) khi cần."""
    chunks = chunks if chunks is not None else load_chunks()
    built: dict = {}
    need_dense = any(m in ("dense", "hybrid_rrf") for m in methods)
    need_bm25 = any(m in ("bm25", "hybrid_rrf") for m in methods)
    dense = DenseRetriever(chunks) if need_dense else None
    bm25 = BM25Retriever(chunks) if need_bm25 else None
    for m in methods:
        if m == "dense":
            built[m] = dense
        elif m == "bm25":
            built[m] = bm25
        elif m == "hybrid_rrf":
            built[m] = HybridRRFRetriever(dense, bm25)
        else:
            raise ValueError(f"Phương pháp không hỗ trợ: {m}")
    return built
