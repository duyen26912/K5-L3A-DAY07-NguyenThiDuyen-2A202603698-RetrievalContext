"""Benchmark retrieval (chưa dùng LLM): chạy các phương pháp trên một split, tính Hit@1, Evidence Hit@3.

Chạy từ thư mục gốc repo (venv đã kích hoạt, Docker pgvector đang chạy):
    python -m src.eval_retrieval --split dev
    python -m src.eval_retrieval --split dev --methods bm25 --top-k 5
    python -m src.eval_retrieval --split dev --no-filter        # bỏ filter chương do người dùng cung cấp

Định nghĩa hit (giống nhau cho mọi phương pháp):
  - Một phần tử gold dạng `evidence` được tìm thấy khi trong danh sách có ít nhất 1 chunk
    cùng source_id + page và norm(evidence) nằm trong norm(chunk.text).
  - Gold dạng `evidence_all` cần MỌI đoạn đều xuất hiện trong các chunk (cùng source_id + page) của danh sách.
  - Các phần tử trong `gold` là evidence thay thế: chỉ cần thỏa 1 phần tử.
  - Hit@k = câu có gold thỏa trong top-k. Mẫu số = số câu có đáp án của split
    (câu thiếu evidence vẫn được chạy và lưu trace, nhưng không tính vào retrieval metrics).
  - Evidence Hit@3 luôn tính trên top-3, kể cả khi --top-k lớn hơn, để so sánh cùng cutoff.

Gold CHỈ được dùng sau khi retrieval xong, để chấm. Query gửi đi là câu hỏi gốc; filter lấy từ trường
`filter` của câu hỏi (thông tin người dùng cung cấp trước), không suy ra từ gold.

Output: results/retrieval/<split>-<method>[-nofilter]-k<top_k>/{summary.json, traces.json}
        và results/retrieval/<split>-comparison[-nofilter]-k<top_k>.md (bảng so sánh).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from src.retrievers import ROOT, build_retrievers, load_chunks, norm

METHODS = ("dense", "bm25", "hybrid_rrf")


def gold_found(results: list[dict], gold: list[dict]) -> bool:
    for g in gold:
        cands = [norm(c["text"]) for c in results
                 if c["source_id"] == g["source_id"] and c["page"] == g["page"]]
        parts = g.get("evidence_all") or [g["evidence"]]
        if all(any(norm(p) in t for t in cands) for p in parts):
            return True
    return False


def first_hit_rank(results: list[dict], gold: list[dict]) -> int | None:
    for k in range(1, len(results) + 1):
        if gold_found(results[:k], gold):
            return k
    return None


def chunk_supports(chunk: dict, gold: list[dict]) -> bool:
    """Chunk này chứa ít nhất một đoạn evidence (để đánh dấu trong trace, không dùng cho metric)."""
    for g in gold:
        if chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"]:
            parts = g.get("evidence_all") or [g["evidence"]]
            if any(norm(p) in norm(chunk["text"]) for p in parts):
                return True
    return False


def run_method(retriever, method: str, questions: list[dict], top_k: int, use_filter: bool) -> list[dict]:
    traces = []
    k_eval = max(top_k, 3)  # luôn lấy đủ top-3 để báo Evidence Hit@3
    for q in questions:
        flt = q.get("filter") if use_filter else None
        trace = {"query_id": q["id"], "split": q["split"], "question": q["question"],
                 "answerable": q.get("answerable", True), "method": method, "top_k": top_k,
                 "filter": flt, "query_sent": q["question"]}
        start = time.perf_counter()
        try:
            results = retriever.search(q["question"], top_k=k_eval, filter=flt)  # không truyền gold
            trace["status"] = "ok"
        except Exception as e:  # ghi lỗi thật vào trace, không thay bằng kết quả giả
            results = []
            trace.update(status="error", error=f"{type(e).__name__}: {e}")
        trace["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)

        slim = [{k: r.get(k) for k in ("rank", "id", "score", "page", "source_id", "dense_rank", "bm25_rank")
                 if r.get(k) is not None} | {"dieu": r["metadata"].get("dieu"),
                                             "chuong": r["metadata"].get("chuong"),
                                             "text": r["text"]}
                for r in results]
        if trace["answerable"]:
            gold = q["gold"]  # chỉ dùng để chấm, sau khi đã có kết quả
            for s, r in zip(slim, results):
                s["supports_gold"] = chunk_supports(r, gold)
            rank = first_hit_rank(results[:top_k], gold)
            trace.update(gold=gold,
                         hit1=gold_found(results[:1], gold),
                         hit3=gold_found(results[:3], gold),
                         hit_at_k=gold_found(results[:top_k], gold),
                         first_hit_rank=rank,
                         reciprocal_rank=round(1 / rank, 4) if rank else 0.0)
        trace["results"] = slim[:max(top_k, 3)]
        traces.append(trace)
    return traces


def summarize(traces: list[dict], method: str, split: str, top_k: int, use_filter: bool,
              init_seconds: float) -> dict:
    ans = [t for t in traces if t["answerable"]]
    n = len(ans)
    errors = [t["query_id"] for t in traces if t["status"] == "error"]
    ms = [t["elapsed_ms"] for t in traces if t["status"] == "ok"]

    def frac(key):
        k = sum(bool(t.get(key)) for t in ans)
        return {"n": k, "N": n, "rate": round(k / n, 4) if n else None}

    return {"method": method, "split": split, "top_k": top_k, "use_filter": use_filter,
            "query": "câu hỏi gốc (không viết lại)",
            "n_questions": len(traces), "n_answerable": n,
            "hit1": frac("hit1"), "evidence_hit3": frac("hit3"), f"hit@{top_k}": frac("hit_at_k"),
            f"mrr@{top_k}": round(sum(t.get("reciprocal_rank", 0) for t in ans) / n, 4) if n else None,
            "misses_at3": [t["query_id"] for t in ans if not t.get("hit3")],
            "errors": errors,
            "median_ms_per_query": round(statistics.median(ms), 2) if ms else None,
            "mean_ms_per_query": round(statistics.mean(ms), 2) if ms else None,
            "init_seconds": round(init_seconds, 2),
            "cost_usd": 0.0, "cost_note": "chạy local (CPU + Docker), không gọi API trả phí"}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Benchmark retrieval trên một split")
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--no-filter", action="store_true", help="bỏ qua trường `filter` của câu hỏi")
    p.add_argument("--questions", type=Path, default=ROOT / "data" / "questions.json")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "retrieval")
    args = p.parse_args()
    use_filter = not args.no_filter

    questions = [q for q in json.loads(args.questions.read_text(encoding="utf-8"))
                 if q["split"] == args.split]  # chỉ đọc đúng split được yêu cầu
    chunks = load_chunks()
    t0 = time.perf_counter()
    retrievers = build_retrievers(args.methods, chunks)
    init_seconds = time.perf_counter() - t0

    tag = ("" if use_filter else "-nofilter") + f"-k{args.top_k}"
    rows = []
    for method in args.methods:
        traces = run_method(retrievers[method], method, questions, args.top_k, use_filter)
        summary = summarize(traces, method, args.split, args.top_k, use_filter, init_seconds)
        out_dir = args.out / f"{args.split}-{method}{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append(summary)
        print(f"[{method}] Hit@1 {summary['hit1']['n']}/{summary['hit1']['N']} | "
              f"Evidence Hit@3 {summary['evidence_hit3']['n']}/{summary['evidence_hit3']['N']} | "
              f"MRR@{args.top_k} {summary[f'mrr@{args.top_k}']} | "
              f"median {summary['median_ms_per_query']} ms | miss@3: {summary['misses_at3']} | "
              f"lỗi: {summary['errors']} -> {out_dir}")

    def cell(f):
        return f"{f['n']}/{f['N']} ({f['rate']:.0%})" if f["N"] else "0/0"

    lines = [f"# Retrieval {args.split} — top_k={args.top_k}, filter={'có' if use_filter else 'không'}", "",
             f"| Phương pháp | Hit@1 | Evidence Hit@3 | Hit@{args.top_k} | MRR@{args.top_k} | median ms/query | Miss@3 | Lỗi |",
             "|---|---|---|---|---|---|---|---|"]
    for s in rows:
        lines.append(f"| {s['method']} | {cell(s['hit1'])} | {cell(s['evidence_hit3'])} | "
                     f"{cell(s[f'hit@{args.top_k}'])} | {s[f'mrr@{args.top_k}']} | {s['median_ms_per_query']} | "
                     f"{', '.join(s['misses_at3']) or '-'} | {len(s['errors'])} |")
    lines += ["", f"Khởi tạo (load model + kết nối DB + index BM25): {init_seconds:.1f} s. Chi phí: 0 USD (local)."]
    table = args.out / f"{args.split}-comparison{tag}.md"
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nBảng so sánh -> {table}")

    for r in retrievers.values():
        if hasattr(r, "close"):
            r.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
