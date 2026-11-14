# Retrieval dev — top_k=3, filter=không

| Phương pháp | Hit@1 | Evidence Hit@3 | Hit@3 | MRR@3 | median ms/query | Miss@3 | Lỗi |
|---|---|---|---|---|---|---|---|
| dense | 7/15 (47%) | 11/15 (73%) | 11/15 (73%) | 0.5889 | 55.61 | d06, d07, d08, d13 | 0 |
| bm25 | 10/15 (67%) | 13/15 (87%) | 13/15 (87%) | 0.7667 | 5.65 | d07, d09 | 0 |
| hybrid_rrf | 11/15 (73%) | 12/15 (80%) | 12/15 (80%) | 0.7667 | 44.56 | d06, d07, d09 | 0 |

Khởi tạo (load model + kết nối DB + index BM25): 20.8 s. Chi phí: 0 USD (local).
