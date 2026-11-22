# Hệ thống Phân tích Cảm xúc TikTok Song ngữ (Tiếng Việt & Tiếng Anh)

Hệ thống **Sentiment Analysis** , kết hợp AI **Gemma 4 E2B** giúp đọc nội dung video ở **Đầu vào** và đánh giá kết quả thực tế **Đầu ra**, toàn bộ lõi phân tích ở giữa dùng **NLP (`underthesea`) + Machine Learning cổ điển + Deep Learning chuyên biệt** cho cả **Tiếng Việt** và **Tiếng Anh** (tự động lọc bỏ các ngôn ngữ khác).

## Cấu trúc thư mục

```text
Sentiment-Analysis/
├── AI_Local/gemma4-e2b/             # Docker Compose + Modelfile Ollama (gemma4-e2b-opt)
├── data/                            # Dữ liệu thu thập thô
│   ├── samples/                     # Bộ dữ liệu mẫu (sample_comments.csv, tiktok.csv)
│   └── <post_id>/                   # Tự tạo theo từng link TikTok: video.mp4, metadata.json, raw_comments.csv
├── outputs/                         # Kết quả cuối cùng sau xử lý theo từng bài đăng
│   └── <post_id>/
│       ├── final_results.csv        # Bình luận sạch (VI/EN) + 2 thang điểm + 7 cảm xúc + ML/DL + Gemma 4
│       ├── filtered_noise.csv       # Tách riêng bình luận nhiễu (tag, URL, spam, ngôn ngữ khác)
│       ├── model_comparison.csv     # Bảng so sánh ML cổ điển, Lexicon, Deep Learning & Hybrid
│       ├── report.json              # Báo cáo JSON đầy đủ 4 giai đoạn
│       └── dashboard.png            # Biểu đồ tổng hợp 2x2
├── models/classical_ml/             # Cache mô hình ML cổ điển tách riêng vi/ và en/
├── src/
│   ├── __init__.py
│   ├── ingestion/                   # Bước 0: Tải video (tương thích máy không có ffmpeg) & crawl comment
│   │   └── tiktok_collector.py
│   ├── preprocessing/               # Giai đoạn 2: Lọc nhiễu, nhận diện VI/EN, underthesea & TF-IDF
│   │   └── text_cleaner.py
│   ├── models/                      # Các mô hình AI, ML & Deep Learning
│   │   ├── media_context.py         # AI Đầu vào: Gemma 4 E2B + Whisper auto-language
│   │   ├── classical_ml.py          # ML cổ điển (NB, LR, SVM, Emotion LR) + Bilingual Lexicon
│   │   ├── sentiment_engine.py      # ViSoBERT/PhoBERT (VI) + Twitter-RoBERTa/DistilRoBERTa (EN) + Thang điểm nhất quán
│   │   └── gemma_auditor.py         # AI Đầu ra: Gemma 4 E2B thẩm định & so sánh mô hình
│   ├── visualization.py             # Vẽ Dashboard 2x2
│   └── pipeline.py                  # Điều phối End-to-End: data/<post_id>/ -> outputs/<post_id>/
├── main.ipynb                       # Notebook chạy trọn gói từ Link TikTok
└── requirements.txt
```

## Cách sử dụng

1. Khởi động Ollama Docker (`gemma4-e2b-opt`):
   ```powershell
   cd E:\A\Sentiment-Analysis\AI_Local\gemma4-e2b
   docker compose up -d
   ```
2. Mở [main.ipynb](main.ipynb) (kernel `hand_env` Python 3.12), nhập link video TikTok và chạy lần lượt các cell.
3. Hoặc chạy bằng CLI:
   ```powershell
   python -m src.pipeline --url "https://www.tiktok.com/@username/video/1234567890"
   ```
