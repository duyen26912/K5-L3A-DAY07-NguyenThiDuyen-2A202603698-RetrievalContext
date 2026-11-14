# Retrieval dev — top_k=3, filter=có

| Phương pháp | Hit@1 | Evidence Hit@3 | Hit@3 | MRR@3 | median ms/query | Miss@3 | Lỗi |
|---|---|---|---|---|---|---|---|
| dense | 8/15 (53%) | 12/15 (80%) | 12/15 (80%) | 0.6556 | 47.47 | d06, d07, d08 | 0 |
| bm25 | 10/15 (67%) | 13/15 (87%) | 13/15 (87%) | 0.7667 | 4.67 | d07, d09 | 0 |
| hybrid_rrf | 11/15 (73%) | 12/15 (80%) | 12/15 (80%) | 0.7667 | 45.41 | d06, d07, d09 | 0 |

Khởi tạo (load model + kết nối DB + index BM25): 33.6 s. Chi phí: 0 USD (local).
