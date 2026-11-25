# Hệ thống Phân tích Cảm xúc TikTok Song ngữ (Tiếng Việt & Tiếng Anh)

Hệ thống **Sentiment Analysis** kết hợp AI **Gemma 4 E2B** giúp đọc ngữ cảnh video/ảnh ở **Đầu vào** (tùy chọn) và thẩm định kết quả thực tế **Đầu ra** (tùy chọn); toàn bộ lõi phân tích ở giữa dùng **NLP (`underthesea`) + Machine Learning cổ điển + Deep Learning chuyên biệt** cho cả **Tiếng Việt** và **Tiếng Anh** (tự động lọc bỏ các ngôn ngữ khác).

## Cấu trúc thư mục

```text
Sentiment-Analysis/
├── AI_Local/gemma4-e2b/             # Docker Compose + Modelfile Ollama (gemma4-e2b-opt)
├── data/                            # Gói dữ liệu chuẩn bị đầu vào theo từng bài đăng
│   └── <post_id>/                   # Tự tạo theo từng link TikTok: video.mp4 (hoặc images/slide_XX.jpg), metadata.json, raw_comments.csv, media_context.json
├── outputs/                         # Kết quả cuối cùng sau xử lý theo từng lượt chạy (6 tệp)
│   └── <run_name>/
│       ├── final_results.csv         # Bảng kết quả dễ đọc cho người xem (12 cột lõi hoặc 16 cột khi bật AI Audit; tiêu đề tiếng Việt 1 dòng, sắp theo lượt thích giảm dần)
│       ├── detailed_model_outputs.csv # Chi tiết kỹ thuật, ID gốc, ngữ cảnh video, xác suất & dự đoán từng mô hình (29 cột lõi hoặc 34 cột khi bật AI Audit, khớp 1-1 theo stt)
│       ├── filtered_noise.csv        # Tách riêng bình luận nhiễu kèm giải thích tiếng Việt (9 cột)
│       ├── model_comparison.csv      # Bảng so sánh ML cổ điển, Lexicon, Deep Learning & Hybrid (tiêu đề tiếng Việt 1 dòng)
│       ├── report.json               # Báo cáo tổng hợp JSON gọn (metadata, ngữ cảnh media, thống kê & so sánh mô hình)
│       └── dashboard.png             # Biểu đồ tổng hợp 2x2
├── models/classical_ml/             # Cache mô hình ML cổ điển tách riêng vi/ và en/
├── src/
│   ├── __init__.py
│   ├── ingestion/                   # Chuẩn bị dữ liệu: Tải video/ảnh Photo Mode (HTTP + Edge), crawl comment + reply, đọc dữ liệu
│   │   └── tiktok_collector.py
│   ├── preprocessing/               # Giai đoạn 2: Lọc nhiễu bảo vệ thực thể video, nhận diện VI/EN, underthesea & TF-IDF ngữ cảnh
│   │   └── text_cleaner.py
│   ├── models/                      # Các mô hình AI, ML & Deep Learning
│   │   ├── media_context.py         # Chuẩn bị ngữ cảnh (Giai đoạn 1): Gemma 4 E2B + Whisper -> lưu/nạp data/<post_id>/media_context.json
│   │   ├── classical_ml.py          # ML cổ điển (NB, LR, SVM, Emotion LR) + Bilingual Lexicon
│   │   ├── sentiment_engine.py      # ViSoBERT/PhoBERT (VI) + Twitter-RoBERTa/DistilRoBERTa (EN) + Context Alignment
│   │   └── gemma_auditor.py         # Giai đoạn 4 (mặc định tắt): Gemma 4 E2B thẩm định ca mỉa mai sâu & so sánh mô hình
│   ├── visualization.py             # Vẽ Dashboard 2x2
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
2. Mở [main.ipynb](main.ipynb), chọn nguồn đầu vào (**`1. Link TikTok`** để chuẩn bị dữ liệu vào `data/<post_id>/`, hoặc **`2. Thư mục đã chuẩn bị sẵn`** `data/<post_id>/` — tự nạp lại `media_context.json` nếu đã có), giữ `RUN_AI_AUDIT = False` (mặc định tắt) hoặc bật `True` nếu muốn thẩm định sâu, rồi chạy lần lượt các cell.
3. Hoặc chạy bằng CLI:
   - **Từ Link TikTok (Giai đoạn 4 mặc định tắt; thêm `--ai-audit` nếu muốn bật):**
     ```powershell
     python -m src.pipeline --url "https://www.tiktok.com/@username/video/1234567890"
     ```
   - **Chỉ chuẩn bị dữ liệu vào `data/<post_id>/` (tải media, crawl comments & lưu `media_context.json` rồi dừng):**
     ```powershell
     python -m src.pipeline --url "https://www.tiktok.com/@username/video/1234567890" --prepare-only
     ```
   - **Từ thư mục đã chuẩn bị sẵn (`data/<post_id>/`, tự dùng lại `media_context.json` cho Giai đoạn 2 & 3, không cần bật Ollama):**
     ```powershell
     python -m src.pipeline --input-dir "data/7711719899852295445"
     ```
   - **Chạy lại kèm thẩm định AI Giai đoạn 4 (cho các ca mỉa mai sâu):**
     ```powershell
     python -m src.pipeline --input-dir "data/7711719899852295445" --ai-audit
     ```
