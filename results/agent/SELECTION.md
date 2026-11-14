# Cấu hình 

Tiêu chí (theo thứ tự ưu tiên):
1. Grounded answer trung bình /2 trên dev (chấm tay)
2. Context Evidence Rate trên dev
3. Từ chối nhầm câu có đáp án (càng ít càng tốt), từ chối đúng câu thiếu evidence
4. Thời gian median / token

Kết quả dev:
| Cấu hình | Grounded | Context Evidence | Từ chối nhầm | Median |
|---|---|---|---|---|
| dense k3 (baseline) | 1,87 | 14/15 | 1/15 | 10,6 s |
| hybrid_rrf k3 + neighbors 1 | 1,73 | 13/15 | 2/15 | 7,8 s |
| bm25 k5 | 1,87 | 14/15 | 1/15 | 6,5 s |

Chọn: bm25, top_k=5, max_searches=3, budget=3000, neighbors=0, có filter, model gemini-3.1-flash-lite.
Lý do: hòa baseline ở tiêu chí 1–3, thắng ở tiêu chí 4.
Sẽ chạy trên test: baseline (dense k3) và cấu hình đã chọn (bm25 k5). Không chỉnh gì sau khi xem test.