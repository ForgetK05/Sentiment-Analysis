# Hệ thống Phân tích Cảm xúc TikTok Song ngữ (Tiếng Việt & Tiếng Anh)

Hệ thống **Sentiment Analysis** kết hợp AI **Gemma 4 E2B** giúp đọc ngữ cảnh video/ảnh ở **Đầu vào** (tùy chọn) và thẩm định kết quả thực tế **Đầu ra** (tùy chọn); toàn bộ lõi phân tích ở giữa dùng **NLP (`underthesea`) + 5 họ Classical Machine Learning v3 (LinearSVC/NBSVM/RBF-SVM, NB/CNB, LR/Ridge/SGD, RF/HistGB + Smart Meta-Ensemble) + Deep Learning chuyên biệt (PhoBERT, ViSoBERT, Twitter-RoBERTa)** cho cả **Tiếng Việt** và **Tiếng Anh**, hỗ trợ chạy linh hoạt qua `--engine hybrid` (mặc định) hoặc `--engine classical` (thuần Classical ML, không nạp PyTorch/Transformers).

## Cấu trúc thư mục

```text
Sentiment-Analysis/
├── AI_Local/gemma4-e2b/             # Docker Compose + Modelfile Ollama (gemma4-e2b-opt)
├── data/                            # Gói dữ liệu chuẩn bị đầu vào theo từng bài đăng
│   └── <post_id>/                   # Tự tạo theo từng link TikTok: video.mp4 (hoặc images/slide_XX.jpg), metadata.json, raw_comments.csv, media_context.json
├── outputs/                         # Kết quả cuối cùng sau xử lý theo từng lượt chạy
│   └── <run_name>/
│       ├── report.html              # Báo cáo tương tác tự chứa (Luồng 4 giai đoạn, 5 họ ML, Ma trận đồng thuận, Comment X-Ray)
│       ├── final_results.csv        # Bảng kết quả dễ đọc cho người xem (12 cột lõi hoặc 16 cột khi bật AI Audit; tiêu đề song ngữ 2 dòng Anh–Việt, sắp theo lượt thích giảm dần)
│       ├── detailed_model_outputs.csv # Chi tiết kỹ thuật, ID gốc, ngữ cảnh video, xác suất & dự đoán từng mô hình đang chạy
│       ├── filtered_noise.csv       # Tách riêng bình luận nhiễu kèm giải thích tiếng Việt (9 cột; tiêu đề song ngữ 2 dòng Anh–Việt)
│       ├── model_comparison.csv     # Bảng so sánh 5 họ Classical ML, Lexicon, Deep Learning & Ensemble kèm timing riêng từng mô hình
│       ├── report.json              # Báo cáo tổng hợp JSON gọn (metadata, pipeline_stages, ngữ cảnh media, thống kê & so sánh mô hình)
│       ├── dashboard_overview.png   # Biểu đồ 2x2 tổng quan khán giả (Polarity, 7 cảm xúc, bám sát video, từ khóa TF-IDF)
│       └── dashboard_models.png     # Biểu đồ 2x2 phân tích & giải thích 5 họ mô hình (Dự đoán, Benchmark F1, Tốc độ, Top Features)
├── models/classical_ml/             # Cache mô hình Classical ML (v3) tách riêng vi/ và en/
├── src/
│   ├── __init__.py
│   ├── ingestion/                   # Chuẩn bị dữ liệu: Tải video/ảnh Photo Mode (HTTP + Edge), crawl comment + reply, đọc dữ liệu
│   │   └── tiktok_collector.py
│   ├── preprocessing/               # Giai đoạn 2: Lọc nhiễu bảo vệ thực thể video, nhận diện VI/EN, underthesea & TF-IDF ngữ cảnh
│   │   └── text_cleaner.py
│   ├── models/                      # Các mô hình AI, Classical ML v3 & Deep Learning
│   │   ├── media_context.py         # Chuẩn bị ngữ cảnh (Giai đoạn 1): Gemma 4 E2B + Whisper -> lưu/nạp data/<post_id>/media_context.json
│   │   ├── classical_ml.py          # Classical ML v3 (Word+Char TF-IDF + 15 Social Features; 3 biến thể SVM, NB/CNB, LR/Ridge/SGD, RF/HistGB + Smart Meta-Ensemble) + Bilingual Lexicon + CLI benchmark
│   │   ├── sentiment_engine.py      # Hỗ trợ 2 chế độ: Hybrid (ML v3 + ViSoBERT/PhoBERT/RoBERTa) hoặc Classical thuần (--engine classical) + Context Alignment
│   │   └── gemma_auditor.py         # Giai đoạn 4 (mặc định tắt): Gemma 4 E2B thẩm định ca mỉa mai sâu & so sánh mô hình
│   ├── visualization.py             # Vẽ Dashboard 2x2 & Benchmark Classical ML v3
│   └── pipeline.py                  # Điều phối End-to-End: data/<post_id>/ -> Giai đoạn 2 & 3 -> outputs/<run_name>/
├── main.ipynb                       # Notebook chạy trọn gói (Link TikTok hoặc Thư mục đã chuẩn bị sẵn; Giai đoạn 4 mặc định tắt)
└── requirements.txt
```

## Tài liệu chi tiết

- [SETUP_LOCAL_AI.md](SETUP_LOCAL_AI.md): Hướng dẫn cài đặt chi tiết **Local AI (`Gemma 4 E2B` qua Docker Ollama)** tối ưu 4 GB VRAM và môi trường `hand_env`.
- [ARCHITECTURE_AND_DATA_GUIDE.md](ARCHITECTURE_AND_DATA_GUIDE.md): Giải thích toàn bộ cấu trúc mã nguồn, luồng hoạt động End-to-End và từ điển đầy đủ các cột CSV/JSON.

## Cách sử dụng

1. *(Chỉ cần khi trích xuất mới `media_context.json` ở Giai đoạn 1 hoặc khi bật Giai đoạn 4 `--ai-audit`)* Khởi động Ollama Docker (`gemma4-e2b-opt`):
   ```powershell
   cd E:\A\Sentiment-Analysis\AI_Local\gemma4-e2b
   docker compose up -d
   ```
2. Mở [main.ipynb](main.ipynb), chọn nguồn đầu vào (**`1. Link TikTok`** để chuẩn bị dữ liệu vào `data/<post_id>/`, hoặc **`2. Thư mục đã chuẩn bị sẵn`** `data/<post_id>/` — tự nạp lại `media_context.json`), rồi chọn ngay trong hộp thoại tương tác:
   - **Giai đoạn 3:** `1` (**ML truyền thống** — Classical ML v3 + Lexicon, 0% Deep Learning) hoặc `2` (**Hybrid hiện tại** — ML v3 + Deep Learning).
   - **Giai đoạn 4:** `y/N` (chọn có chạy **Gemma 4 E2B** thẩm định mỉa mai sâu hay không, và phạm vi `1. Chỉ ca khó (uncertain)` / `2. Toàn bộ (all)` — notebook hỏi cho cả Nguồn 1 và Nguồn 2).
3. Hoặc chạy bằng CLI:
   - **Từ Link TikTok (Giai đoạn 4 mặc định tắt; thêm `--ai-audit` nếu muốn bật):**
     ```powershell
     python -m src.pipeline --url "https://www.tiktok.com/@username/video/1234567890"
     ```
   - **Chỉ chuẩn bị dữ liệu vào `data/<post_id>/` (tải media, crawl comments & lưu `media_context.json` rồi dừng):**
     ```powershell
     python -m src.pipeline --url "https://www.tiktok.com/@username/video/1234567890" --prepare-only
     ```
   - **Từ thư mục đã chuẩn bị sẵn ở Giai đoạn 3 chế độ 2 — Hybrid mặc định (`--engine 2` hoặc `--engine hybrid`):**
     ```powershell
     python -m src.pipeline --input-dir "data/7711719899852295445" --engine 2
     ```
   - **Chạy Giai đoạn 3 chế độ 1 — Thuần ML truyền thống (`--engine 1` hoặc `--engine classical` — 0% Deep Learning; thêm `--full-models` nếu muốn chạy đủ 10+1 mô hình thay vì 5 đại diện + Ensemble):**
     ```powershell
     python -m src.pipeline --input-dir "data/7711719899852295445" --engine 1
     ```
   - **Huấn luyện lại cache Classical ML v3 (5 họ ML cổ điển + OOF Meta-Ensemble + 6 ML Emotion) cho VI/EN:**
     ```powershell
     python -m src.models.classical_ml --lang vi en --retrain
     ```
   - **Chạy kèm thẩm định AI Giai đoạn 4 (`--ai-audit`; thêm `--audit-scope uncertain` để chỉ chấm các ca khó):**
     ```powershell
     python -m src.pipeline --input-dir "data/7711719899852295445" --ai-audit --audit-scope uncertain
     ```
