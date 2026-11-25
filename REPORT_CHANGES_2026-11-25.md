# Báo cáo Tổng hợp Thay đổi Dự án — Ngày 25/11/2026

- **Dự án:** Hệ thống Phân tích Cảm xúc TikTok Song ngữ Việt – Anh (`e:\A\Sentiment-Analysis`)
- **Ngày thực hiện:** `2026-11-25`
- **Commit chính trong ngày:** `7b74bb82e13fc2bfdaf95c3795933bf565aa7308` (`"update system"`, 47 tệp thay đổi, `+21158 / -3121` dòng tính cả dữ liệu & cache)
- **Thay đổi chưa commit (working tree):** Cập nhật `MAX_COMMENTS = 2000` trong `main.ipynb`

---

## 1. Tóm tắt 5 nhóm thay đổi chính

### 1.1. Crawl đầy đủ bình luận trả lời (Reply) & đồng bộ metadata
- **`src/ingestion/tiktok_collector.py`**:
  - Bật mặc định `include_replies=True` và `max_replies_per_comment=None` (lấy toàn bộ reply cho tới khi chạm `max_comments`) ở `_download_photo_post_playwright`, `collect_tiktok_post`, `_collect_comments_sync`, `crawl_tiktok_comments` và `_scroll_and_collect_comments_on_page`.
  - Thêm hàm `_expand_visible_replies()`: tự động tìm và bấm các nút mở rộng reply trên DOM theo regex tiếng Việt và tiếng Anh (`_VIEW_REPLIES_TEXT_RE`), đồng thời loại trừ các nút thu gọn / hành động (`_HIDE_OR_ACTION_TEXT_RE`: `Ẩn`, `Hide`, `Thu gọn`, `Collapse`, `Trả lời`, `Reply`). Có cơ chế chống lặp vô hạn theo từng thread (`data-agy-reply-stalls >= 2` và `consecutive_empty >= 6`).
  - Nâng cấp `_parse_comment_items()` và thêm `_build_comment_row()` để bóc cả gói `/api/comment/list/reply/` lẫn mảng `reply_comment` preview trong `/api/comment/list/`, trích xuất thêm 4 cột quan hệ hội thoại: `is_reply`, `parent_comment_id`, `reply_to_comment_id`, `reply_to_username` (nâng `raw_comments.csv` từ 8 lên 12 cột `_COMMENT_CSV_COLUMNS`).
  - Nâng cấp `_extract_dom_fallback()` để quét cả node cấp 2 (`comment-level-2`) và dùng `_make_comment_adder()` nâng cấp dòng fallback bằng dữ liệu API mà không tạo bản ghi trùng lặp.
  - Ép kiểu chuỗi (`_TABULAR_ID_DTYPES`) cho các cột ID 19 chữ số khi đọc CSV/Excel và chuyển đổi boolean an toàn qua `_parse_bool()`.
- **`src/models/sentiment_engine.py`**:
  - Bổ sung 2 thuộc tính `is_reply: bool = False` và `parent_comment_id: str = ""` vào `AnalyzedComment` và điền dữ liệu từ `metadata_rows` trong `CommentAnalyzer.analyze_comments()`.

### 1.2. Nâng cấp Giai đoạn 1 (AI Đầu vào) & ngữ cảnh Giai đoạn 4
- **`src/models/media_context.py`**:
  - Nâng chất lượng hình ảnh & cấu hình suy luận mặc định trong `MediaAnalyzer.__init__`: `image_size = 768` (từ `512`), `max_frames = 8` (từ `10`), `temperature = 0.1` (từ `0.15`), nén JPEG chất lượng `90` (từ `85`), truyền `num_ctx = 16384` và `repeat_penalty = 1.1` vào `_call_ollama_chat()`.
  - Thêm `_natural_sort_key()` để sắp xếp ảnh slideshow đúng thứ tự tự nhiên (`slide_1.jpg`, `slide_2.jpg`, ..., `slide_10.jpg`).
  - Tách `_resize_frame_bgr()` và `_stamp_frame_label()`: resize ảnh trước rồi mới đóng dấu nhãn mốc thời gian (`F1 00:02`) hoặc số thứ tự ảnh (`Anh 1/N`) để nhãn luôn sắc nét.
  - Cải thiện `extract_video_frames()`: bỏ 3% đầu (fade-in) và 5% cuối (outro logo TikTok) với video > 30 frame, đọc timestamp thực từ `CAP_PROP_POS_MSEC` (dự phòng `idx / fps`), và bổ sung nhánh đọc tuần tự chống tràn RAM (`_read_video_sequential_safe()`) cho video VFR hoặc thiếu metadata.
  - Bổ sung chế độ tự động **2-pass** (`enable_two_pass_slideshow = True`, `two_pass_batch_size = 4`, `SLIDE_BATCH_SCHEMA`) khi bài slideshow có nhiều hơn `max_frames` ảnh.
  - Cải thiện `transcribe_audio()`: thêm `_clean_whisper_prompt()` loại bỏ URL, `#hashtag`, `@mention`, emoji trước khi mồi `initial_prompt`; bật `condition_on_previous_text=False`, `no_speech_threshold=0.6`, `compression_ratio_threshold=2.4`, lọc `no_speech_prob < 0.7`, giữ định dạng mốc thời gian `[MM:SS-MM:SS]`; thêm `unload_whisper_model()` để giải phóng RAM sau Giai đoạn 1.
  - Nâng cấp `MEDIA_RESPONSE_SCHEMA` (11 trường theo luồng **Evidence-First**: `chu_tren_man_hinh`, `dien_bien_hinh_anh`, `chi_tiet_va_luan_diem` → `tom_tat_ngan` → 7 trường phân loại), cập nhật `_SYSTEM_PROMPT`, bổ sung 3 trường danh sách vào `MediaAnalysisResult`, và sửa `_parse_json_response()` không đổ chuỗi JSON lỗi vào `tom_tat_ngan`.
- **`src/models/gemma_auditor.py`**:
  - Cập nhật `_format_video_context()` để truyền thêm tối đa 5 ý từ `chi_tiet_va_luan_diem` sang Giai đoạn 4 và thêm `num_ctx=16384`, `repeat_penalty=1.05` trong `_call_ollama_json()`.

### 1.3. Đầu vào linh hoạt từ File/Thư mục & tùy chọn bật/tắt Giai đoạn 1 / Giai đoạn 4
- **`src/ingestion/tiktok_collector.py` & `requirements.txt`**:
  - Mở rộng `load_comments_dataframe()` hỗ trợ đọc trực tiếp **`.csv`**, **`.xlsx` / `.xls`** (thêm `openpyxl>=3.1.0` vào `requirements.txt`), **`.json`**, **`.txt`**, tự tách bỏ dòng chú thích ở tiêu đề 2 dòng và thêm `"original_text"` vào `COMMENT_COLUMN_CANDIDATES`.
- **`src/pipeline.py`**:
  - Thêm `resolve_local_input_dir()` và tham số `input_dir` (CLI `--input-dir`) để tự nhận diện file bình luận, video/ảnh và `caption` trong một thư mục cục bộ.
  - Thêm `run_media_analysis` (CLI `--skip-media`) để có thể bỏ qua Giai đoạn 1 ngay cả khi có media; khi chỉ truyền file bình luận, pipeline tự bỏ qua Bước 0 & Giai đoạn 1.
  - Thêm cặp cờ CLI `--ai-audit` / `--no-ai-audit` (giữ `--skip-audit` làm alias) để người dùng bật/tắt Giai đoạn 4 (`run_ai_audit`), không gọi Ollama khi không cần thẩm định.
  - Sau Giai đoạn 1 (khi `unload_ollama_after_stage1=True`), gọi cả `unload_ollama_model()` và `unload_whisper_model()`.

### 1.4. Tái cấu trúc Output CSV thành 3 tầng rõ ràng (`src/pipeline.py`)
Tách riêng phần kết quả hữu ích khỏi các thuộc tính kỹ thuật phụ (không làm mất bất kỳ trường dữ liệu nào):
- **`cleaned_final_results.csv`** (`build_cleaned_final_results_df`, sắp theo `likes` giảm dần):
  - **10 cột lõi**: `username`, `is_reply`, `likes`, `created_at`, `language`, `original_text`, `polarity_vi`, `score_5`, `score_10`, `emotion_vi`.
  - **+4 cột AI** (chỉ xuất hiện khi bật Giai đoạn 4): `gemma_polarity_vi`, `gemma_score_5`, `gemma_is_sarcasm`, `gemma_reason`.
- **`final_results.csv`** (`format_final_results_df`, giữ tiêu đề 2 dòng Anh–Việt):
  - **18 cột hữu ích**: `comment_id`, `is_reply`, `parent_comment_id`, `username`, `likes`, `reply_count`, `created_at`, `language`, `original_text`, `cleaned_text`, `polarity_label`, `polarity_vi`, `polarity_confidence`, `score_5`, `score_10`, `rating_level_vi`, `emotion_vi`, `emotion_confidence`.
  - **+6 cột AI** (chỉ xuất hiện khi bật Giai đoạn 4): `gemma_polarity`, `gemma_emotion_vi`, `gemma_score_5`, `gemma_is_sarcasm`, `gemma_agrees_polarity`, `gemma_reason`.
- **`detailed_model_outputs.csv`** *(mới — `build_detailed_model_outputs_df`)*:
  - **20 cột kỹ thuật**: `comment_id`, `username`, `original_text`, `nickname`, `segmented_text`, `is_duplicate`, `has_sarcasm_signal`, `needs_audit`, `polarity_score`, `score_POS`, `score_NEG`, `score_NEU`, `emotion_label`, `pred_nb`, `pred_lr`, `pred_svm`, `pred_lexicon`, `pred_visobert`, `pred_dl_primary`, `emotion_ml_label`.
  - **+3 cột AI bổ sung** (chỉ xuất hiện khi bật Giai đoạn 4): `gemma_audited`, `gemma_emotion`, `gemma_agrees_emotion`.
- **`model_comparison.csv`**: tự động loại bỏ các cột `*_vs_gemma*` đang rỗng khi tắt Giai đoạn 4.

### 1.5. Đồng bộ Notebook, Exports & Tài liệu dự án
- **`main.ipynb`**: Cập nhật cả 6 cell hỗ trợ chọn nguồn đầu vào (`1: Link TikTok | 2: File nguồn | 3: Thư mục có sẵn`), hỏi bật/tắt Giai đoạn 4 (`RUN_AI_AUDIT`), truyền đầy đủ tham số mới và hiển thị động các cột output để không gặp `KeyError` khi tắt AI Audit.
- **`src/__init__.py`**: Export bổ sung `build_cleaned_final_results_df`, `build_detailed_model_outputs_df`, `format_final_results_df`, `resolve_local_input_dir`.
- **`README.md`** & **`ARCHITECTURE_AND_DATA_GUIDE.md`**: Cập nhật toàn bộ sơ đồ luồng, từ điển cột mới của `raw_comments.csv`, `final_results.csv`, `cleaned_final_results.csv`, `detailed_model_outputs.csv`, thông số Giai đoạn 1 và bảng tham số CLI/Notebook.

---

## 2. Chi tiết tệp mã nguồn & tài liệu đã sửa (Commit `7b74bb8` + working tree)

| Tệp | Thay đổi chính (từ `git log --stat` & `git diff`) |
| :--- | :--- |
| [tiktok_collector.py](file:///e:/A/Sentiment-Analysis/src/ingestion/tiktok_collector.py) | `+ / -` tổng 621 dòng: tự động mở reply, schema 12 cột, DOM fallback cấp 2, đọc `.xlsx/.xls` & tiêu đề 2 dòng |
| [media_context.py](file:///e:/A/Sentiment-Analysis/src/models/media_context.py) | `+ / -` tổng 540 dòng: 768px, 8 frame, nhãn timestamp, chống OOM, 2-pass slideshow, Whisper lọc ảo giác, schema 11 trường |
| [pipeline.py](file:///e:/A/Sentiment-Analysis/src/pipeline.py) | `+ / -` tổng 342 dòng: tách 3 tầng CSV, `resolve_local_input_dir()`, tùy chọn Giai đoạn 1 & Giai đoạn 4, CLI flags mới |
| [gemma_auditor.py](file:///e:/A/Sentiment-Analysis/src/models/gemma_auditor.py) | `+5` dòng: truyền `chi_tiet_va_luan_diem` vào ngữ cảnh và cấu hình `num_ctx`/`repeat_penalty` |
| [sentiment_engine.py](file:///e:/A/Sentiment-Analysis/src/models/sentiment_engine.py) | `+13` dòng: thêm `is_reply`, `parent_comment_id` vào `AnalyzedComment` |
| [src/__init__.py](file:///e:/A/Sentiment-Analysis/src/__init__.py) | `+ / -` 17 dòng: lazy export thêm 4 hàm mới từ `src.pipeline` |
| [requirements.txt](file:///e:/A/Sentiment-Analysis/requirements.txt) | `+1` dòng: thêm `openpyxl>=3.1.0` |
| [main.ipynb](file:///e:/A/Sentiment-Analysis/main.ipynb) | Cập nhật 6 cell điều phối tương tác; thêm thay đổi chưa commit đổi `MAX_COMMENTS = 200` → `2000` |
| [README.md](file:///e:/A/Sentiment-Analysis/README.md) | `+ / -` 56 dòng: cập nhật cấu trúc thư mục và ví dụ CLI/Notebook |
| [ARCHITECTURE_AND_DATA_GUIDE.md](file:///e:/A/Sentiment-Analysis/ARCHITECTURE_AND_DATA_GUIDE.md) | `+ / -` 294 dòng: cập nhật kiến trúc, Giai đoạn 1/4, từ điển 3 tầng CSV & CLI |
| Dữ liệu mẫu / chạy thử | Xóa bộ mẫu cũ `data/7666481155348237576/`; thêm kết quả chạy thật trên `data/7711719899852295445/` và `outputs/7711719899852295445/` |

---

## 3. Kết quả kiểm thử & chạy thực tế hôm nay

1. **Kiểm thử tự động offline (trong môi trường `hand_env`):**
   - Bộ test logic reply & CSV ([verify_replies.py](file:///C:/Users/kiend/.gemini/antigravity/brain/5c176fa9-e738-4164-b454-47ff64cc8ff1/scratch/verify_replies.py)) chạy trước khi tái cấu trúc 44 → 24 cột đạt `ALL_CHECKS_PASSED`.
   - Bộ test tích hợp toàn hệ thống ([verify_all.py](file:///C:/Users/kiend/.gemini/antigravity/brain/5c176fa9-e738-4164-b454-47ff64cc8ff1/scratch/verify_all.py)) đạt `ALL_PROJECT_VERIFICATIONS_PASSED` (xác minh `MediaAnalyzer`, 2-pass slideshow mock, đọc `.csv`/`.json`/`.txt`, `resolve_local_input_dir`, đúng số cột của `cleaned_final_results.csv` / `final_results.csv` / `detailed_model_outputs.csv`, chạy `SentimentPipeline.run()` với `run_ai_audit=False` xuất đủ 7 tệp đầu ra, kiểm tra CLI & cú pháp `main.ipynb`).
2. **Kết quả chạy thực tế trên video TikTok `7711719899852295445`:**
   - **Bước 0 (`data/7711719899852295445/raw_comments.csv`):** Thu thập được **200 bình luận**, trong đó có **110 bình luận chính và 90 reply** (`is_reply=True`).
   - **Giai đoạn 1 (`outputs/7711719899852295445/report.json`):** Gemma 4 E2B trích xuất được chủ đề cụ thể: *"So sánh lịch sử phát triển và năng lực công nghệ giữa Việt Nam và các siêu cường tư bản."*
   - **Giai đoạn 2 & 3:** Lọc `3` bình luận nhiễu (`FILTERED_UNSUPPORTED_LANG`), giữ `197` bình luận hợp lệ (`1` bình luận trùng lặp).
   - **Giai đoạn 4 & Output:** Chạy với tùy chọn bỏ qua thẩm định AI (`ai_audit_verdict` là `null`), xuất đầy đủ 7 tệp trong `outputs/7711719899852295445/`: `cleaned_final_results.csv`, `final_results.csv`, `detailed_model_outputs.csv`, `filtered_noise.csv`, `model_comparison.csv`, `report.json`, và `dashboard.png`.

---

## 4. Các điểm lưu ý hiện tại

- **Chưa commit ở `main.ipynb`:** Biến `MAX_COMMENTS` vừa được sửa từ `200` lên `2000`. Tuy nhiên, tham số `max_scrolls` trong `tiktok_collector.py` vẫn đang mặc định là `40` lần cuộn (chưa truyền từ `pipeline.py` / `main.ipynb`), nên với bài nhiều bình luận, vòng cuộn có thể dừng ở lần cuộn thứ 40 trước khi đạt 2000 bình luận.
- **Đọc file Excel (`.xlsx`/`.xls`):** Gói `openpyxl` đã được thêm vào `requirements.txt` nhưng chưa cài trong môi trường `hand_env` tại thời điểm kiểm tra (`openpyxl_installed: False`), cần chạy `pip install openpyxl` nếu muốn dùng file Excel trực tiếp.
