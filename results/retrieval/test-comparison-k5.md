# Retrieval test — top_k=5, filter=có

| Phương pháp | Hit@1 | Evidence Hit@3 | Hit@5 | MRR@5 | median ms/query | Miss@3 | Lỗi |
|---|---|---|---|---|---|---|---|
| bm25 | 3/5 (60%) | 4/5 (80%) | 5/5 (100%) | 0.7167 | 2.5 | t02 | 0 |

Khởi tạo (load model + kết nối DB + index BM25): 0.0 s. Chi phí: 0 USD (local).
