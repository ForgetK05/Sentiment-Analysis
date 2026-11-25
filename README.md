# Hệ thống Phân tích Cảm xúc TikTok Song ngữ (Tiếng Việt & Tiếng Anh)

Hệ thống **Sentiment Analysis** kết hợp AI **Gemma 4 E2B** giúp đọc ngữ cảnh video/ảnh ở **Đầu vào** (tùy chọn) và thẩm định kết quả thực tế **Đầu ra** (tùy chọn); toàn bộ lõi phân tích ở giữa dùng **NLP (`underthesea`) + Machine Learning cổ điển + Deep Learning chuyên biệt** cho cả **Tiếng Việt** và **Tiếng Anh** (tự động lọc bỏ các ngôn ngữ khác).

## Cấu trúc thư mục

```text
Sentiment-Analysis/
├── AI_Local/gemma4-e2b/             # Docker Compose + Modelfile Ollama (gemma4-e2b-opt)
├── data/                            # Dữ liệu thu thập thô hoặc file nguồn do người dùng cung cấp
│   ├── samples/                     # Bộ dữ liệu mẫu (sample_comments.csv)
│   └── <post_id>/                   # Tự tạo theo từng link TikTok: video.mp4 (hoặc images/slide_XX.jpg), metadata.json, raw_comments.csv
├── outputs/                         # Kết quả cuối cùng sau xử lý theo từng lượt chạy
│   └── <run_name>/
│       ├── cleaned_final_results.csv # Bản tóm gọn 10 cột cốt lõi (hoặc 14 cột khi bật AI Audit, sắp theo likes giảm dần)
│       ├── final_results.csv         # Các thuộc tính phân tích hữu ích (18 cột lõi, hoặc 24 cột khi bật AI Audit) + chú thích tiếng Việt
│       ├── detailed_model_outputs.csv # Thuộc tính kỹ thuật, xác suất & dự đoán riêng của từng mô hình (20 cột, hoặc 23 cột khi bật AI Audit)
│       ├── filtered_noise.csv        # Tách riêng bình luận nhiễu (tag, URL, spam, ngôn ngữ khác)
│       ├── model_comparison.csv      # Bảng so sánh ML cổ điển, Lexicon, Deep Learning & Hybrid
│       ├── report.json               # Báo cáo JSON đầy đủ 4 giai đoạn
│       └── dashboard.png             # Biểu đồ tổng hợp 2x2
├── models/classical_ml/             # Cache mô hình ML cổ điển tách riêng vi/ và en/
├── src/
│   ├── __init__.py
│   ├── ingestion/                   # Bước 0: Tải video/ảnh Photo Mode (HTTP + Edge), crawl comment + reply, đọc CSV/Excel/JSON/TXT
│   │   └── tiktok_collector.py
│   ├── preprocessing/               # Giai đoạn 2: Lọc nhiễu, nhận diện VI/EN, underthesea & TF-IDF
│   │   └── text_cleaner.py
│   ├── models/                      # Các mô hình AI, ML & Deep Learning
│   │   ├── media_context.py         # AI Đầu vào: Gemma 4 E2B (768px, timestamp, Evidence-First 11 trường) + Whisper
│   │   ├── classical_ml.py          # ML cổ điển (NB, LR, SVM, Emotion LR) + Bilingual Lexicon
│   │   ├── sentiment_engine.py      # ViSoBERT/PhoBERT (VI) + Twitter-RoBERTa/DistilRoBERTa (EN) + Thang điểm nhất quán
│   │   └── gemma_auditor.py         # AI Đầu ra: Gemma 4 E2B thẩm định & so sánh mô hình (tùy chọn bật/tắt)
│   ├── visualization.py             # Vẽ Dashboard 2x2
│   └── pipeline.py                  # Điều phối End-to-End: data/<post_id>/ (hoặc file/folder cục bộ) -> outputs/<run_name>/
├── main.ipynb                       # Notebook chạy trọn gói (hỗ trợ Link TikTok / File CSV-XLSX-JSON-TXT / Thư mục & bật/tắt Giai đoạn 4)
└── requirements.txt
```

## Tài liệu chi tiết

- [SETUP_LOCAL_AI.md](SETUP_LOCAL_AI.md): Hướng dẫn cài đặt chi tiết **Local AI (`Gemma 4 E2B` qua Docker Ollama)** tối ưu 4 GB VRAM và môi trường `hand_env`.
- [ARCHITECTURE_AND_DATA_GUIDE.md](ARCHITECTURE_AND_DATA_GUIDE.md): Giải thích toàn bộ cấu trúc mã nguồn, luồng hoạt động End-to-End và từ điển đầy đủ các cột CSV/JSON.

## Cách sử dụng

1. *(Nếu dùng Giai đoạn 1 phân tích video/ảnh hoặc Giai đoạn 4 thẩm định bằng AI)* Khởi động Ollama Docker (`gemma4-e2b-opt`):
   ```powershell
   cd E:\A\Sentiment-Analysis\AI_Local\gemma4-e2b
   docker compose up -d
   ```
2. Mở [main.ipynb](main.ipynb), chọn nguồn đầu vào (`Link TikTok`, `File nguồn CSV/XLSX/JSON/TXT`, hoặc `Thư mục đã tải`) và chọn có chạy Giai đoạn 4 (`AI Audit`) hay không, rồi chạy lần lượt các cell.
3. Hoặc chạy bằng CLI:
   - **Từ Link TikTok (mặc định crawl cả reply & bật AI Audit):**
     ```powershell
     python -m src.pipeline --url "https://www.tiktok.com/@username/video/1234567890"
     ```
   - **Từ file nguồn trực tiếp (CSV / Excel / JSON / TXT), không chạy Giai đoạn 4:**
     ```powershell
     python -m src.pipeline --comments "data/samples/sample_comments.csv" --no-ai-audit
     ```
   - **Từ file nguồn kèm video/ảnh cục bộ (chạy đủ cả Giai đoạn 1 và tắt Giai đoạn 4):**
     ```powershell
     python -m src.pipeline --comments "comments.xlsx" --media "video.mp4" --caption "Mô tả video" --no-ai-audit
     ```
   - **Từ thư mục đã tải sẵn (`data/<post_id>/`):**
     ```powershell
     python -m src.pipeline --input-dir "data/7711458816386190614" --no-ai-audit
     ```
