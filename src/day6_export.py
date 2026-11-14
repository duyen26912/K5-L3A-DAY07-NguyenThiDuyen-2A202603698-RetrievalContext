"""Bước 1: kết nối vector store Day 6, xuất chunk + benchmark sang schema Day 7 và kiểm tra.

Chạy từ thư mục gốc repo:
    python -m src.day6_export --day6-questions <đường dẫn Day 6>/benchmark/questions.jsonl

Việc script làm:
  1. Đọc toàn bộ chunk của strategy Day 6 đã chọn (mặc định B) từ pgvector.
  2. Ghi data/chunks.json theo schema Day 7 (id, text, source_id, source, page, metadata).
  3. Chuyển benchmark Day 6 (JSONL) sang data/questions.json với trường `gold`.
  4. Kiểm tra evidence của từng câu có nằm trọn trong một chunk đúng nguồn + trang không
     (schema Day 7 yêu cầu vậy); câu nào không thỏa thì in ra để sửa tay.
  5. Kiểm tra query embedding tương thích index: embed lại text của vài chunk bằng model
     hiện tại, so cosine với vector đã lưu (phải ~1.0), rồi chạy thử một dense query.
  6. Ghi data/day6_connection.json (model, số chiều, strategy, số chunk, kết quả kiểm tra).

Không có fallback sang vector giả: nếu DB hoặc model lỗi, script dừng và báo lỗi.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def norm(s: str) -> str:
    """Chuẩn hóa để so khớp evidence: NFC, gộp khoảng trắng (PDF hay ngắt dòng giữa câu)."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "")).strip()


def fetch_chunks(conn: psycopg.Connection, strategy: str) -> list[dict]:
    rows = conn.execute(
        "SELECT chunk_id, document_id, source, page, chuong, muc, dieu, text, metadata "
        "FROM chunks WHERE strategy = %s ORDER BY page, chunk_id", (strategy,)).fetchall()
    if not rows:
        raise RuntimeError(f"Không có chunk nào cho strategy={strategy!r}; kiểm tra volume Day 6")
    out = []
    for chunk_id, doc_id, source, page, chuong, muc, dieu, text, meta in rows:
        meta = dict(meta or {})
        meta.update({"strategy": strategy, "chuong": chuong, "muc": muc, "dieu": dieu})
        out.append({"id": chunk_id, "text": text, "source_id": doc_id, "source": source,
                    "page": page, "metadata": meta})
    return out


def _gold(source_id, page, dieu, evidence) -> dict:
    """evidence dạng chuỗi -> `evidence`; dạng list -> `evidence_all` (phải có ĐỦ mọi đoạn)."""
    g = {"source_id": source_id, "page": page, "dieu": dieu}
    if isinstance(evidence, list):
        g["evidence_all"] = evidence
    else:
        g["evidence"] = evidence
    return g


def convert_questions(path: Path) -> list[dict]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        item = {"id": q["id"], "split": q["split"], "question": q["question"],
                "answerable": True, "expected_behavior": "search_and_answer",
                "expected_answer": q["expected_answer"],
                "gold": [_gold(q["gold_source_id"], q["gold_page"], q.get("gold_dieu"), q["evidence_quote"])]
                        + [_gold(a["source_id"], a["page"], a.get("dieu"), a["evidence_quote"])
                           for a in q.get("alt_gold", [])],  # các phần tử gold là evidence thay thế
                "type": q.get("type")}
        if q.get("filter"):
            item["filter"] = q["filter"]  # filter người dùng cung cấp sẵn trong câu hỏi (Day 6)
            item["filter_context"] = q.get("filter_context")
        questions.append(item)
    return questions


def check_evidence(questions: list[dict], chunks: list[dict]) -> list[dict]:
    by_page: dict[tuple, list[dict]] = {}
    for c in chunks:
        by_page.setdefault((c["source_id"], c["page"]), []).append(c)
    report = []
    for q in questions:
        for g in q["gold"]:
            cands = by_page.get((g["source_id"], g["page"]), [])
            parts = g.get("evidence_all") or [g["evidence"]]
            per_part = [[c["id"] for c in cands if norm(ev) in norm(c["text"])] for ev in parts]
            report.append({"id": q["id"], "page": g["page"], "evidence_chunks": per_part,
                           "ok": all(per_part)})
    return report


def check_embedding(conn, strategy: str, model_name: str, chunks: list[dict]) -> dict:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(model_name, device="cpu")
    dim = int(model.get_sentence_embedding_dimension())
    stored_dim = conn.execute(
        "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
    ).fetchone()[0]
    if dim != stored_dim:
        raise RuntimeError(f"Model cho {dim} chiều nhưng index lưu vector({stored_dim})")

    sample = [chunks[i] for i in (0, len(chunks) // 2, len(chunks) - 1)]
    sims = []
    for c in sample:
        stored = conn.execute("SELECT embedding::text FROM chunks WHERE strategy=%s AND chunk_id=%s",
                              (strategy, c["id"])).fetchone()[0]
        sv = np.array(json.loads(stored), dtype=np.float32)
        nv = model.encode([c["text"]], normalize_embeddings=True)[0]
        sims.append(float(np.dot(sv, nv) / (np.linalg.norm(sv) * np.linalg.norm(nv))))

    probe = "Thời giờ làm việc bình thường tối đa bao nhiêu giờ một ngày?"
    qv = model.encode([probe], normalize_embeddings=True)[0]
    vec = "[" + ",".join(f"{x:.7f}" for x in qv) + "]"
    top = conn.execute(
        "SELECT chunk_id, page, 1 - (embedding <=> %s::vector) AS score FROM chunks "
        "WHERE strategy=%s ORDER BY embedding <=> %s::vector, chunk_id LIMIT 3",
        (vec, strategy, vec)).fetchall()
    return {"model": model_name, "dimensions": dim, "stored_dimensions": stored_dim,
            "reembed_cosine_sample": [round(s, 6) for s in sims],
            "probe_query": probe,
            "probe_top3": [{"id": r[0], "page": r[1], "score": round(float(r[2]), 4)} for r in top]}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--day6-questions", required=True, type=Path)
    p.add_argument("--strategy", default=os.getenv("PG_STRATEGY", "B"))
    p.add_argument("--out", default=ROOT / "data", type=Path)
    args = p.parse_args()

    db_url = os.getenv("DATABASE_URL")
    model_name = os.getenv("EMBEDDING_MODEL")
    if not db_url or not model_name:
        raise SystemExit("Thiếu DATABASE_URL hoặc EMBEDDING_MODEL; copy .env.example thành .env")

    with psycopg.connect(db_url, connect_timeout=5) as conn:
        counts = dict(conn.execute("SELECT strategy, count(*) FROM chunks GROUP BY strategy").fetchall())
        chunks = fetch_chunks(conn, args.strategy)
        questions = convert_questions(args.day6_questions)
        evidence = check_evidence(questions, chunks)
        emb = check_embedding(conn, args.strategy, model_name, chunks)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (args.out / "questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    info = {"vector_store": "PostgreSQL 16 + pgvector 0.8.7 (Docker, exact cosine search, không ANN index)",
            "table": "chunks", "strategy": args.strategy, "chunks_per_strategy": counts,
            "pages": len({c["page"] for c in chunks}), "embedding": emb,
            "evidence_check": {"ok": sum(e["ok"] for e in evidence), "total": len(evidence),
                               "failed": [e for e in evidence if not e["ok"]]}}
    (args.out / "day6_connection.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Strategy {args.strategy}: {len(chunks)} chunks, {info['pages']} trang -> {args.out / 'chunks.json'}")
    print(f"Câu hỏi: {len(questions)} ({sum(q['split'] == 'dev' for q in questions)} dev / "
          f"{sum(q['split'] == 'test' for q in questions)} test) -> {args.out / 'questions.json'}")
    print(f"Embedding: {emb['model']} | {emb['dimensions']}d (index {emb['stored_dimensions']}d) | "
          f"cosine embed lại: {emb['reembed_cosine_sample']}")
    print("Dense probe top-3:", emb["probe_top3"])
    print(f"Evidence nằm trọn trong 1 chunk đúng trang: {info['evidence_check']['ok']}/{len(evidence)}")
    for e in info["evidence_check"]["failed"]:
        print(f"  - {e['id']} (trang {e['page']}): evidence bị cắt qua nhiều chunk hoặc lệch text -> sửa tay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
