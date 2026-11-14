# Retrieval test — top_k=3, filter=có

| Phương pháp | Hit@1 | Evidence Hit@3 | Hit@3 | MRR@3 | median ms/query | Miss@3 | Lỗi |
|---|---|---|---|---|---|---|---|
| dense | 3/5 (60%) | 4/5 (80%) | 4/5 (80%) | 0.7 | 23.46 | t02 | 0 |
| bm25 | 3/5 (60%) | 4/5 (80%) | 4/5 (80%) | 0.6667 | 2.92 | t02 | 0 |
| hybrid_rrf | 4/5 (80%) | 5/5 (100%) | 5/5 (100%) | 0.9 | 25.93 | - | 0 |

Khởi tạo (load model + kết nối DB + index BM25): 9.0 s. Chi phí: 0 USD (local).
