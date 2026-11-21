# Hệ thống Phân tích Nội dung TikTok & Cảm xúc Bình luận

Hệ thống gồm 2 giai đoạn tách biệt, tối ưu cho máy có **4GB VRAM (RTX 3050 Laptop)**:

1. **Giai đoạn 1 — Phân tích Video/Ảnh bằng Gemma 4 E2B (`gemma4-e2b-opt`)**
   - Hỗ trợ video (có hoặc không có âm thanh), ảnh đơn hoặc nhiều ảnh (Photo Mode).
   - Tự động kiểm tra luồng âm thanh bằng `av` và bóc lời thoại tiếng Việt bằng `faster-whisper` (chạy CPU `int8`).
   - Tổng hợp khung hình + lời thoại + caption thành JSON gồm: `chu_de_chinh`, `quan_diem_noi_dung`, `tom_tat_ngan`, `tu_khoa`.
2. **Giai đoạn 2 — Phân loại Bình luận thuần Deep Learning (`5CD-AI/Vietnamese-Sentiment-visobert`)**
   - Sử dụng **ViSoBERT** fine-tune sẵn cho tiếng Việt mạng xã hội (giữ nguyên emoji, chuẩn hóa nhẹ teencode).
   - Gán nhãn 3 lớp: `POS` (**Tích cực**), `NEG` (**Tiêu cực**), `NEU` (**Trung tính**) kèm xác suất và tổng hợp xu hướng chung.

## Cấu trúc thư mục

```text
Sentiment-Analysis/
├── AI_Local/gemma4-e2b/     # Cấu hình Ollama Docker cho gemma4-e2b-opt
├── data/
│   └── sample_comments.csv  # 20 comment TikTok mẫu tiếng Việt
├── src/
│   ├── __init__.py
│   ├── media_analyzer.py    # Giai đoạn 1: OpenCV + faster-whisper + Ollama Gemma 4
│   ├── comment_classifier.py# Giai đoạn 2: Tiền xử lý + ViSoBERT
│   └── pipeline.py          # Ghép 2 giai đoạn, CLI, xuất JSON/CSV và vẽ biểu đồ
├── phase1.ipynb             # Notebook chạy thử và trực quan hóa
├── crawl_tiktok_comments.ipynb # Notebook thu thập comment từ link TikTok ra CSV
└── requirements.txt
```

## Hướng dẫn cài đặt

1. **Khởi động Ollama Docker (`gemma4-e2b-opt`):**
   ```powershell
   cd E:\A\Sentiment-Analysis\AI_Local\gemma4-e2b
   docker compose up -d
   ```

2. **Tạo môi trường Python (khuyên dùng Python 3.11 hoặc 3.12) và cài thư viện:**
   ```powershell
   cd E:\A\Sentiment-Analysis
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

## Cách sử dụng

### 1. Thu thập comment từ link TikTok (tùy chọn)
Nếu chưa có sẵn file comment, mở [crawl_tiktok_comments.ipynb](crawl_tiktok_comments.ipynb) (cùng cấp với `phase1.ipynb`):
1. Đảm bảo đã cài `playwright` (`pip install -r requirements.txt` hoặc `pip install playwright`); notebook dùng trực tiếp **Microsoft Edge** có sẵn trên Windows (`channel="msedge"`) nên không cần tải thêm Chromium.
2. Dán link video/ảnh TikTok vào `TIKTOK_URL` và chạy các cell.
3. Notebook xuất file `data/tiktok_comments_<post_id>.csv` (cột đầu tiên `comment`, chuẩn `utf-8-sig`) để bạn gán thẳng vào `COMMENTS_PATH` trong `phase1.ipynb`.

### 2. Chạy phân tích bằng Jupyter Notebook
Mở [phase1.ipynb](phase1.ipynb), chọn kernel `.venv`, chỉnh `MEDIA_PATH` tới video/ảnh của bạn (và `COMMENTS_PATH` nếu dùng file CSV vừa thu thập) rồi chạy lần lượt từng cell.

### 3. Chạy bằng dòng lệnh (CLI)
- Chạy thử nhanh phần phân loại comment trên dữ liệu mẫu:
  ```powershell
  python -m src.pipeline --comments data/sample_comments.csv --output outputs --save-chart outputs/chart.png
  ```
- Chạy đầy đủ cả Video/Ảnh + Comment:
  ```powershell
  python -m src.pipeline --media "G:\Download\Download.mp4" --comments data/sample_comments.csv --caption "Caption bài đăng" --output outputs --save-chart outputs/chart.png
  ```
- Nếu video không có âm thanh hoặc muốn bỏ qua bước nhận dạng giọng nói:
  ```powershell
  python -m src.pipeline --media "video.mp4" --comments data/sample_comments.csv --no-audio --output outputs
  ```
