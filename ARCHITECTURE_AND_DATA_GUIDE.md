# Tài liệu Giải thích Chi tiết Cấu trúc, Cách hoạt động & Từ điển Dữ liệu Dự án

> **Đối tượng dự án:** Hệ thống Phân tích Cảm xúc Bình luận TikTok Song ngữ (`vi` / `en`) Đa phương thức (`e:\A\Sentiment-Analysis`).  
> **Nguyên tắc kiến trúc cốt lõi:** Chỉ dùng mô hình AI tạo sinh (**Gemma 4 E2B**) tại **Đầu vào** (đọc ngữ cảnh video/ảnh) và **Đầu ra** (thẩm định thực tế & tổng kết); toàn bộ phần lõi ở giữa chạy bằng **NLP (`underthesea`) + Học máy cổ điển + Luật từ điển + Học sâu chuyên biệt** trên CPU.

---

## Mục lục

- [1. Bản đồ cấu trúc toàn bộ dự án](#1-bản-đồ-cấu-trúc-toàn-bộ-dự-án)
- [2. Luồng hoạt động từ đầu đến cuối (End-to-End Flow)](#2-luồng-hoạt-động-từ-đầu-đến-cuối-end-to-end-flow)
- [3. Giải thích chi tiết từng thành phần mã nguồn (`src/`)](#3-giải-thích-chi-tiết-từng-thành-phần-mã-nguồn-src)
- [4. Từ điển dữ liệu đầu vào & mô hình lưu đệm (`data/` và `models/`)](#4-từ-điển-dữ-liệu-đầu-vào--mô-hình-lưu-đệm-data-và-models)
- [5. Từ điển dữ liệu kết quả đầu ra (`outputs/<post_id>/`)](#5-từ-điển-dữ-liệu-kết-quả-đầu-ra-outputspost_id)
- [6. Bảng tra cứu tham số cấu hình & vận hành (`main.ipynb` & CLI)](#6-bảng-tra-cứu-tham-số-cấu-hình--vận-hành-mainipynb--cli)

---

## 1. Bản đồ cấu trúc toàn bộ dự án

### 1.1. Cây thư mục tổng thể

```text
e:\A\Sentiment-Analysis\
├── AI_Local/gemma4-e2b/                  # Hạ tầng AI cục bộ chạy qua Docker Ollama
│   ├── docker-compose.yml                # Cấu hình container GPU, bộ nhớ KV Cache 4GB VRAM
│   ├── Modelfile                         # Định nghĩa mô hình tùy biến `gemma4-e2b-opt`
│   ├── ollama_models/                    # Lưu trữ trọng số GGUF của Gemma 4 E2B trên ổ E:
│   └── test.ipynb                        # Notebook kiểm thử kết nối Ollama độc lập
│
├── data/                                 # KHO DỮ LIỆU THU THẬP THÔ (Đầu vào)
│   ├── samples/                          # Dữ liệu mẫu tĩnh (sample_comments.csv)
│   └── <post_id>/                        # Tự động sinh theo từng link TikTok
│       ├── video.mp4 (hoặc images/)      # Video gốc tải về (hoặc bộ ảnh Photo Mode)
│       ├── metadata.json                 # Siêu dữ liệu bài đăng (caption, likes, duration...)
│       └── raw_comments.csv              # Toàn bộ bình luận thô thu thập từ trình duyệt
│
├── models/classical_ml/                  # BỘ ĐỆM MÔ HÌNH HỌC MÁY CỔ ĐIỂN (Cache .joblib)
│   ├── vi/                               # Nhánh Tiếng Việt: NB, LR, Linear SVM, Emotion LR
│   │   ├── polarity_models.joblib
│   │   ├── emotion_ml_model.joblib
│   │   └── benchmark_metrics.json
│   └── en/                               # Nhánh Tiếng Anh: NB, LR, Linear SVM, Emotion LR
│       ├── polarity_models.joblib
│       ├── emotion_ml_model.joblib
│       └── benchmark_metrics.json
│
├── outputs/                              # KHO KẾT QUẢ CUỐI CÙNG SAU XỬ LÝ (Đầu ra)
│   └── <post_id>/                        # Khớp mã bài đăng của thư mục trong data/
│       ├── final_results.csv             # 42 cột chi tiết từng bình luận sạch (kèm chú thích tiếng Việt)
│       ├── cleaned_final_results.csv     # Bản tóm gọn 15 cột quan trọng nhất (sắp theo likes giảm dần)
│       ├── filtered_noise.csv            # 8 cột tách riêng bình luận nhiễu/ngôn ngữ khác
│       ├── model_comparison.csv          # 15 cột so sánh đối chiếu 8 mô hình/nhánh
│       ├── report.json                   # Toàn bộ dữ liệu tổng hợp dạng JSON phân cấp
│       └── dashboard.png                 # Biểu đồ 2x2 trực quan hóa kết quả
│
├── src/                                  # MÃ NGUỒN CHÍNH (Package Python)
│   ├── __init__.py                       # Xuất biểu tượng (Lazy Export) cho toàn gói src
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── tiktok_collector.py           # Bước 0: Tải video/ảnh & crawl bình luận TikTok
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── text_cleaner.py               # Giai đoạn 2: Lọc nhiễu, nhận diện VI/EN, underthesea, TF-IDF
│   ├── models/
│   │   ├── __init__.py
│   │   ├── media_context.py              # Giai đoạn 1: OpenCV + PyAV/Whisper + Gemma 4 E2B (Input)
│   │   ├── classical_ml.py               # Giai đoạn 3A: Từ điển song ngữ + NB/LR/SVM + Benchmark
│   │   ├── sentiment_engine.py           # Giai đoạn 3B: PhoBERT/ViSoBERT/RoBERTa + Hybrid + Thang điểm
│   │   └── gemma_auditor.py              # Giai đoạn 4: Gemma 4 E2B (Output) thẩm định & so sánh mô hình
│   ├── visualization.py                  # Vẽ bảng điều khiển dashboard.png (2x2)
│   └── pipeline.py                       # Điều phối End-to-End & Giao diện dòng lệnh CLI
│
├── main.ipynb                            # Notebook 6 ô chạy 1-click từ link TikTok
├── requirements.txt                      # Danh sách thư viện Python yêu cầu
├── README.md                             # Hướng dẫn cài đặt và sử dụng nhanh
└── Bao_Cao_Du_An_Sentiment_Analysis.docx # Báo cáo đồ án chuẩn học thuật Word (.docx)
```

### 1.2. Giải thích cấu hình hạ tầng AI cục bộ (`AI_Local/gemma4-e2b/`)

Hai tệp cấu hình trong `AI_Local/gemma4-e2b/` quyết định việc mô hình đa phương thức **Gemma 4 E2B** chạy ổn định trên card đồ họa laptop giới hạn **4 GB VRAM** (NVIDIA RTX 3050):

| Tệp | Tham số / Biến | Giá trị | Ý nghĩa kỹ thuật & tác dụng thực tế |
| :--- | :--- | :--- | :--- |
| `docker-compose.yml` | `ports` | `"11434:11434"` | Mở cổng API REST chuẩn của Ollama để mã Python (`requests`) gọi tại `http://localhost:11434`. |
| `docker-compose.yml` | `volumes` | `E:/.../ollama_models:/root/.ollama` | Lưu trọng số mô hình trực tiếp lên ổ `E:`, giữ nguyên mô hình khi khởi động lại container. |
| `docker-compose.yml` | `OLLAMA_FLASH_ATTENTION` | `1` | Bật thuật toán **Flash Attention**, giảm lượng VRAM tiêu thụ của phép tính Attention khi nạp chuỗi nhiều khung hình ảnh. |
| `docker-compose.yml` | `OLLAMA_KV_CACHE_TYPE` | `q8_0` | Lượng tử hóa bộ nhớ đệm ngữ cảnh (KV Cache) xuống 8-bit, giúp giảm khoảng 40% VRAM mà gần như không giảm độ chính xác. |
| `docker-compose.yml` | `OLLAMA_MAX_LOADED_MODELS` | `1` | Chỉ cho phép nạp tối đa 1 mô hình vào VRAM tại một thời điểm, ngăn lỗi tràn bộ nhớ GPU. |
| `docker-compose.yml` | `OLLAMA_KEEP_ALIVE` | `10m` | Tự động giữ mô hình trên VRAM trong 10 phút giữa các batch liên tiếp ở Giai đoạn 4 để tránh nạp lại nhiều lần. |
| `Modelfile` | `FROM gemma4:e2b` | `gemma4:e2b` | Kế thừa ảnh mô hình nền đa phương thức Gemma 4 E2B của Google. |
| `Modelfile` | `PARAMETER num_ctx` | `16384` | Mở rộng cửa sổ ngữ cảnh từ 4.096 mặc định lên **16.384 tokens** (đủ chứa tối đa 10 khung hình video + lời thoại Whisper + Prompt), nhưng không quá lớn để làm tràn 4 GB VRAM. |
| `Modelfile` | `PARAMETER num_gpu` | `99` | Ép đẩy toàn bộ các lớp mạng của Gemma 4 E2B lên GPU rời để đạt tốc độ suy luận cao nhất. |
| `Modelfile` | `temperature / top_p / top_k` | `1.0 / 0.95 / 64` | Giá trị mặc định ở cấp Modelfile; khi gọi API ở Giai đoạn 1 và Giai đoạn 4, code Python ghi đè xuống `0.15` và `0.10` để AI trả lời chuẩn xác theo JSON Schema. |

---

## 2. Luồng hoạt động từ đầu đến cuối (End-to-End Flow)

### 2.1. Sơ đồ luồng dữ liệu tổng thể

```mermaid
flowchart TD
    U["Người dùng nhập Link TikTok hoặc dữ liệu cục bộ<br/>(main.ipynb hoặc CLI python -m src.pipeline)"] --> S0

    subgraph B0["Bước 0: Thu thập dữ liệu thô (src/ingestion/tiktok_collector.py)"]
        S0["download_tiktok_media(): yt-dlp tải video / Playwright tải Photo Mode"] --> D1["Lưu data/&lt;post_id&gt;/video.mp4 &amp; metadata.json"]
        S0 --> S0B["crawl_tiktok_comments(): Playwright Edge mở trang &amp; cuộn bình luận"]
        S0B --> D2["Lưu data/&lt;post_id&gt;/raw_comments.csv (8 cột)"]
    end

    D1 --> G1A
    subgraph S1["Giai đoạn 1: AI Đầu vào — Ngữ cảnh đa phương thức (GPU 4GB + CPU)"]
        G1A["MediaAnalyzer.analyze(): Trích tối đa 10 khung hình (OpenCV)"] --> G1C["Gemma 4 E2B (gemma4-e2b-opt, GPU)<br/>xuất MediaAnalysisResult (JSON 8 trường)"]
        G1B["PyAV tách tiếng 16kHz + Faster-Whisper (small, CPU)<br/>chép lời tự nhận diện VI/EN"] --> G1C
        G1C --> G1D["unload_ollama_model(): Gửi keep_alive=0<br/>Giải phóng 100% VRAM GPU"]
    end

    D2 --> G2A
    G1D --> G2A

    subgraph S2["Giai đoạn 2: Tiền xử lý NLP &amp; Lọc nhiễu (100% CPU, không dùng LLM)"]
        G2A["normalize_tiktok_emojis() + clean_comment_text()<br/>Chuẩn hóa Emoji [Cười...] &amp; Teencode"] --> G2B["detect_language(): Phân loại lai VI / EN / OTHER"]
        G2B --> G2C["_detect_filter_reason() + underthesea POS/NER<br/>Lọc 7 loại nhiễu"]
        G2C -->|Không hợp lệ| NOISE["Nhánh nhiễu: filtered_noise.csv (8 cột)"]
        G2C -->|Hợp lệ vi/en| G2D["underthesea word_tokenize (vi) + TF-IDF Keywords<br/>+ Gắn cờ is_duplicate, has_sarcasm_signal"]
    end

    G2D --> R1
    subgraph S3["Giai đoạn 3: Lõi Phân tích Cảm xúc Song ngữ (100% CPU)"]
        R1{"Định tuyến ngôn ngữ<br/>vi hay en?"}
        R1 -->|vi| VI_ML["ML cổ điển (NB, LR, SVM, Emo-LR) + Lexicon VI"]
        R1 -->|vi| VI_DL["Deep Learning VI: PhoBERT + ViSoBERT + PhoBERT-Emotion"]
        R1 -->|en| EN_ML["ML cổ điển EN (NB, LR, SVM, Emo-LR) + Lexicon EN"]
        R1 -->|en| EN_DL["Deep Learning EN: Twitter-RoBERTa + DistilRoBERTa-Emotion"]
        VI_ML --> HYB["Hybrid Ensemble + Emotion Cues + compute_dual_scale()<br/>Tạo AnalyzedComment (35 thuộc tính)"]
        VI_DL --> HYB
        EN_ML --> HYB
        EN_DL --> HYB
    end

    HYB --> G4A
    G1C -. "Truyền JSON Ngữ cảnh 8 trường" .-> G4A

    subgraph S4["Giai đoạn 4: AI Đầu ra — Thẩm định thực tế (GPU 4GB)"]
        G4A["GemmaAuditor.audit_comments():<br/>Gemma 4 E2B chấm độc lập theo lô 8 câu + ngữ cảnh"] --> G4B["build_model_comparison():<br/>Agreement %, Macro-F1, Cohen's Kappa, score_5 MAE"]
        G4B --> G4C["generate_overall_verdict():<br/>Nhận định độ chính xác, điểm mạnh/yếu"]
    end

    NOISE --> EXP
    G4C --> EXP["SentimentPipeline._export_outputs() + plot_sentiment_dashboard()"]
    EXP --> OUT["outputs/&lt;post_id&gt;/ :<br/>final_results.csv | cleaned_final_results.csv | filtered_noise.csv<br/>model_comparison.csv | report.json | dashboard.png"]
```

### 2.2. Chu kỳ quản lý tài nguyên (CPU ↔ GPU 4GB)

Để chạy trơn tru đồng thời **Gemma 4 E2B (VLM)**, **Faster-Whisper** và tối đa **5 mô hình Transformer (PhoBERT/RoBERTa)** trên laptop 4 GB VRAM, đường ống điều phối thứ tự tải tài nguyên theo đúng chu kỳ 3 pha:
1. **Pha 1 — Chiếm GPU (`Giai đoạn 1`):** Ollama nạp `gemma4-e2b-opt` lên GPU 4 GB để đọc 10 khung hình + transcript Whisper và sinh JSON ngữ cảnh. Song song đó, `faster-whisper` bản `small` chạy bằng `int8` trên **CPU** nên không tranh VRAM. Kết thúc pha, `unload_ollama_model()` gửi `"keep_alive": 0` để đẩy toàn bộ trọng số Gemma ra khỏi VRAM.
2. **Pha 2 — Thuần CPU (`Giai đoạn 2 & 3`):** Toàn bộ xử lý `underthesea`, trích TF-IDF, bộ mô hình Scikit-learn (`.joblib`), bộ luật từ điển và các mô hình HuggingFace (`torch`, `device="cpu"`) chạy trên RAM + CPU đa nhân. Các mô hình Transformer được **tải lười (lazy load)** theo đúng ngôn ngữ xuất hiện (ví dụ video toàn tiếng Anh sẽ không nạp 3 mô hình tiếng Việt).
3. **Pha 3 — Chiếm lại GPU (`Giai đoạn 4`):** Nếu `run_ai_audit=True`, Ollama nạp lại `gemma4-e2b-opt` lên GPU để chấm độc lập theo từng lô 8 bình luận và viết đánh giá tổng kết.

### 2.3. Truy vết thực tế một bình luận xuyên suốt 4 giai đoạn (Data Trace)

Lấy đúng 1 bình luận thật từ lần chạy video `7709458511838612744` (`"Phim Việt thoại sảng phết 2026: Chị Chị Em Em 3 - Phần 1"`):

1. **Bước 0 (`data/7709458511838612744/raw_comments.csv`):**
   - `comment_id = "7710022854226035457"`, `username = "19036388441017tech"`, `nickname = "Babydon'tcry"`, `likes = 153`, `reply_count = 11`, `created_at = "2026-11-20 06:06:49"`.
   - `comment` gốc: `"Thời phong kiến đã có trao nhẫn cầu hôn rồi à mn lại còn trao nhẫn nữa"`.
2. **Giai đoạn 2 (`PreprocessedComment`):**
   - `detect_language` phát hiện ký tự đặc trưng tiếng Việt (`ơ, ê, à, ầ, ô, ữ`) → `language = "vi"`.
   - `clean_comment_text` thay teencode `"mn"` thành `"mọi người"` → `cleaned_text = "Thời phong kiến đã có trao nhẫn cầu hôn rồi à mọi người lại còn trao nhẫn nữa"`.
   - Kiểm tra qua 7 tầng lọc nhiễu → vượt qua hết → `is_valid = True`, `filter_reason = "VALID"`.
   - `underthesea.word_tokenize` nối từ ghép → `segmented_text = "Thời phong_kiến đã có trao nhẫn cầu_hôn rồi à mọi người lại còn trao_nhẫn nữa"`.
   - Không trùng lặp (`is_duplicate = False`), không có emoji/cụm cà khịa (`has_sarcasm_signal = False`).
3. **Giai đoạn 3 (`AnalyzedComment`):**
   - Định tuyến vào nhánh **Tiếng Việt**:
     + `TF-IDF + NB` → `NEG`, `TF-IDF + LR` → `POS`, `TF-IDF + Linear SVM` → `POS`, `Lexicon` → `NEU`, `ViSoBERT` → `NEG`, `PhoBERT` → `POS`, `Emotion ML` → `Disgust`.
     + `PhoBERT Emotion` dự đoán → `emotion_label = "Surprise"` (`emotion_vi = "Ngạc nhiên"`, `emotion_confidence = 0.9666`).
     + Cộng trọng số Hybrid `0.40 PhoBERT + 0.20 ViSoBERT + 0.25 SVM + 0.15 Lexicon` → `score_POS = 0.5683`, `score_NEG = 0.2687`, `score_NEU = 0.1629` → `polarity_label = "POS"`, `polarity_confidence = 0.5683`.
     + `compute_dual_scale` tính `polarity_score = +0.2996`, `score_5 = 4`, `score_10 = 6.50`, `rating_level_vi = "Tích cực"`.
     + Vì `PhoBERT (POS) != ViSoBERT (NEG)` và `polarity_confidence = 0.5683 < 0.65` → gắn cờ `needs_audit = True`.
4. **Giai đoạn 4 (`GemmaAuditItem` & dòng số 2 của `outputs/7709458511838612744/final_results.csv`):**
   - Gemma 4 E2B đọc bình luận kèm ngữ cảnh video review phim phong kiến → đánh giá câu này thực chất chỉ đang nhận xét trung lập về chi tiết cốt truyện → `gemma_polarity = "NEU"`, `gemma_score_5 = 3`, `gemma_emotion = "Other"` (`"Trung tính / Khác"`), `gemma_is_sarcasm = False`, `gemma_agrees_polarity = False`, `gemma_agrees_emotion = False`, `gemma_reason = "Đây là một nhận xét mang tính chất suy ngẫm về cốt truyện/tình tiết."`.

---

## 3. Giải thích chi tiết từng thành phần mã nguồn (`src/`)

### 3.1. `src/__init__.py` — Cơ chế xuất biểu tượng lười (Lazy Export)

- Khai báo danh sách công khai `__all__` (36 lớp, hàm và hằng số) cùng từ điển ánh xạ `_EXPORT_MAP`.
- Dùng hàm `__getattr__(name)`: khi người dùng gọi `from src import SentimentPipeline`, Python mới tự động nạp đúng mô-đun con (`src.pipeline`), tránh tình trạng vừa `import src` đã phải kéo theo `torch`, `transformers`, `cv2` gây chậm khởi động.

### 3.2. `src/ingestion/tiktok_collector.py` (714 dòng) — Bước 0: Thu thập dữ liệu

| Thành phần | Loại | Vai trò & logic hoạt động |
| :--- | :--- | :--- |
| `TikTokMediaBundle` | `dataclass` (13 trường) | Đóng gói kết quả tải về gồm `url`, `resolved_url`, `post_id`, `media_type` (`"video"` hoặc `"images"`), `media_paths`, `caption`, `hashtags`, `uploader`, `title`, `duration`, `like_count`, `comment_count`, `metadata_path`. |
| `extract_post_id(url)` | Hàm | Dùng biểu thức chính quy trích chuỗi số ID cuối URL TikTok (`/video/<id>` hoặc `/photo/<id>`). Nếu không khớp thì tạo mã băm `post_<hash>` để đặt tên thư mục. |
| `_resolve_short_url(url)` | Hàm nội bộ | Gửi HTTP GET cho các link rút gọn như `vt.tiktok.com` / `vm.tiktok.com` để lấy đường dẫn đầy đủ chứa `@username/video/<id>`. |
| `_build_ytdlp_options(outtmpl, has_ffmpeg)` | Hàm nội bộ | Nếu `shutil.which("ffmpeg")` có FFmpeg, cho phép `yt-dlp` tải và ghép luồng tốt nhất (`bv*+ba/b`); nếu máy **không có FFmpeg**, ép chọn luồng MP4 ghép sẵn (`b[ext=mp4]/best[ext=mp4]/b`) để tránh lỗi thiếu FFmpeg. |
| `download_tiktok_media(url, data_dir)` | Hàm chính | Giải quyết short link, tạo thư mục `data/<post_id>/`, chạy `yt-dlp` tải `video.mp4`. Nếu gặp bài đăng dạng trượt ảnh (Photo Mode), tự chuyển sang `_download_photo_post_playwright()` để lưu tối đa 8 ảnh `.jpg`/`.webp` vào `data/<post_id>/images/`, rồi ghi `metadata.json`. |
| `_parse_comment_items(payload)` | Hàm nội bộ | Đọc JSON từ API ẩn `api/comment/list` của TikTok mà Playwright bắt được trên mạng: bóc `cid`, `text`, `digg_count` (likes), `reply_comment_total`, `create_time`, `unique_id`, `nickname` (và bình luận trả lời lồng nếu bật `include_replies`). |
| `_extract_dom_fallback(page)` | Hàm dự phòng | Nếu API mạng không bắt đủ, đọc trực tiếp các thẻ DOM HTML trên trang TikTok (`[data-e2e='comment-item']`...) bằng JavaScript thực thi trên trình duyệt. |
| `crawl_tiktok_comments(...)` | Hàm chính | Mở trình duyệt Microsoft Edge thật (`channel="msedge"`, `headless=False`), tạm dừng để người dùng vượt popup đăng nhập/CAPTCHA (nếu có), sau đó tự động cuộn tối đa `max_scrolls=40` lần (dừng sớm nếu đủ `max_comments` hoặc 4 lần cuộn liên tiếp không có thêm bình luận mới). Ghi kết quả ra `data/<post_id>/raw_comments.csv`. |
| `load_comments_dataframe(source)` | Hàm tiện ích | Đọc linh hoạt `.csv`, `.json` hoặc `.txt`, chuẩn hóa tên cột văn bản về `"comment"` và điền mặc định các cột metadata (`comment_id`, `username`, `nickname`, `likes`, `reply_count`, `created_at`, `post_url`). |

### 3.3. `src/models/media_context.py` (577 dòng) — Giai đoạn 1: AI Đầu vào (Gemma 4 E2B + Whisper)

| Thành phần | Loại | Vai trò & logic hoạt động |
| :--- | :--- | :--- |
| `MEDIA_RESPONSE_SCHEMA` | `dict` (JSON Schema) | Ép Ollama trả về đúng cấu trúc JSON 8 trường bắt buộc: `chu_de_chinh`, `muc_dich_chinh`, `quan_diem_noi_dung`, `doi_tuong_chinh` (mảng chuỗi), `sac_thai_noi_dung` (enum 5 giá trị: `Tích cực`, `Tiêu cực`, `Châm biếm / Mỉa mai`, `Tranh cãi`, `Trung lập`), `tom_tat_ngan`, `tu_khoa` (mảng 4–8 từ) và `luu_y_danh_gia_comment`. |
| `MediaAnalysisResult` | `dataclass` (15 trường) | Lưu toàn bộ 8 trường ngữ cảnh + thông tin kỹ thuật (`media_type`, `source_paths`, `num_frames_used`, `has_audio`, `caption`, `transcript`, `raw_response`). |
| `MediaAnalyzer.extract_video_frames(...)` | Phương thức | Mở video bằng `cv2.VideoCapture`, chia đều tổng số frame để lấy tối đa `max_frames=10` khung hình, thu nhỏ cạnh dài về tối đa `672px` (`max_image_dim=672`) và mã hóa Base64 JPEG. |
| `MediaAnalyzer._decode_audio_16k_mono(...)` | Phương thức | Dùng `av` (**PyAV**) đọc luồng audio từ `.mp4` và resample sang mảng `float32` đơn kênh `16.000 Hz` trực tiếp trong RAM, không cần cài FFmpeg. |
| `MediaAnalyzer.transcribe_audio(...)` | Phương thức | Nạp `WhisperModel("small", device="cpu", compute_type="int8")`, bật bộ lọc khoảng lặng `vad_filter=True`, đặt `language=None` để tự động phát hiện tiếng Việt hoặc tiếng Anh, truyền caption vào `initial_prompt` làm mồi nhận diện tên riêng. |
| `MediaAnalyzer.unload_ollama_model()` | Phương thức | Gửi HTTP POST tới `/api/generate` với `"keep_alive": 0` để buộc Ollama đẩy Gemma 4 E2B khỏi VRAM ngay sau Giai đoạn 1. |
| `MediaAnalyzer.analyze(...)` | Phương thức chính | Kết hợp ảnh Base64 + lời thoại Whisper + caption theo `_SYSTEM_PROMPT` (ưu tiên đối chiếu chữ trên màn hình OCR ngữ nghĩa với caption), gọi `/api/chat` (`temperature=0.15`, `format=MEDIA_RESPONSE_SCHEMA`) và phân tích JSON trả về. |

### 3.4. `src/preprocessing/text_cleaner.py` (678 dòng) — Giai đoạn 2: NLP & Lọc nhiễu song ngữ

| Thành phần | Loại | Vai trò & logic hoạt động |
| :--- | :--- | :--- |
| `PreprocessedComment` | `dataclass` (10 trường) | Chứa `original_text`, `cleaned_text`, `segmented_text`, `ml_text`, `language` (`vi` \| `en` \| `other`), `tokens`, `is_valid`, `filter_reason`, `is_duplicate`, `has_sarcasm_signal`. |
| `normalize_tiktok_emojis(text)` | Hàm | Chuyển hơn 25 mã emoji chữ của TikTok như `[Cười chảy nước mắt]`, `[Khóc]`, `[Yêu]`, `[Phẫn nộ]` về đúng ký tự Unicode (`😂`, `😭`, `😍`, `😡`). |
| `detect_language(text)` | Hàm 4 tầng | 1) Có chữ ngoài Latin (Nga, Trung, Nhật, Hàn, Ả Rập...) → `other`; 2) Có từ chức năng/ký tự riêng Tây Ban Nha/Bồ Đào Nha/Pháp nhưng thiếu dấu riêng tiếng Việt → `other`; 3) Có dấu riêng Việt (`ă, â, đ, ê, ô, ơ, ư`, thanh hỏi/ngã/nặng...) hoặc ≥ 2 từ chức năng Việt → `vi`; 4) Chữ ASCII / có từ chức năng tiếng Anh → `en`. |
| `clean_comment_text(...)` | Hàm | Chuẩn hóa Unicode NFKC, xóa khoảng trắng thừa; với tiếng Việt thay thế hơn 40 từ **teencode an toàn** (`ko`/`k` → `không`, `dc`/`đc` → `được`, `mn` → `mọi người`, `tr` → `trời`...) và chủ động giữ nguyên các từ dễ gây hiểu nhầm (`bt`, `v`, `t`). |
| `_safe_word_tokenize / _safe_pos_tag / _safe_ner` | Hàm bọc `underthesea` | Gọi thư viện `underthesea` cho tiếng Việt (`word_tokenize(..., format="text")` để nối từ ghép bằng `_`), với tiếng Anh tách từ đơn giản giữ lại nguyên âm và emoji. |
| `_is_name_only_comment(...)` | Hàm | Dùng `pos_tag` (`Np`) và `ner` (`PER`) kiểm tra xem toàn bộ câu có phải chỉ là tên người hay không. Có 3 lá chắn chống lọc nhầm: bỏ qua chuỗi viết hoa toàn bộ (`ALL-CAPS` như `"HOLY TRANSITION"`), yêu cầu ít nhất 2 từ chữ cái, và kiểm tra không chứa từ ngữ cảm xúc theo token nguyên vẹn. |
| `_detect_filter_reason(...)` | Hàm | Quyết định mã lọc: `FILTERED_EMPTY`, `FILTERED_UNSUPPORTED_LANG`, `FILTERED_URL_OR_SYMBOL`, `FILTERED_TAG_FRIEND`, `FILTERED_NAME_ONLY`, `FILTERED_SPAM_AD`, `FILTERED_MEANINGLESS`, hoặc `"VALID"`. |
| `_has_sarcasm_signal(...)` | Hàm | Gắn `True` nếu phát hiện dấu hiệu nói mỉa như `cười muốn bịnh`, `cười xỉu`, `diễn sâu`, `hề hước`, hoặc từ khen đi kèm emoji `😂`/`🤣`/`🙄`/`🤡`. |
| `TextPreprocessor.process_batch(comments)` | Phương thức | Xử lý hàng loạt tất cả bình luận, chuẩn hóa và đánh dấu cờ trùng lặp `is_duplicate` (không phân biệt chữ hoa/thường sau khi đã thấy bản đầu tiên). |
| `extract_keywords_tfidf(...)` | Hàm | Dùng `TfidfVectorizer(ngram_range=(1, 2))` trên tập `ml_text` (đã loại stopword song ngữ Việt–Anh) để lấy Top 10 từ khóa cho 3 nhóm: `overall`, `positive`, `negative`. |

### 3.5. `src/models/classical_ml.py` (886 dòng) — Giai đoạn 3A: ML cổ điển & Bộ từ điển song ngữ

| Thành phần | Loại | Vai trò & logic hoạt động |
| :--- | :--- | :--- |
| `BilingualLexiconScorer` | Lớp luật & từ điển | Chấm điểm cảm xúc dựa trên từ điển dương/âm riêng cho `vi` (`_VI_POSITIVE_LEXICON`, `_VI_NEGATIVE_LEXICON`) và `en` (`_EN_POSITIVE_LEXICON`, `_EN_NEGATIVE_LEXICON`). Áp dụng phủ định trong cửa sổ 2 từ trước (`không`, `chẳng`, `not`, `never` → nhân `-0.85`), tăng/giảm mức độ (`quá`, `rất`, `freaking` → nhân `1.45`; `hơi`, `slightly` → nhân `0.70`), và nhận diện cấu trúc leo thang tiếng Việt như `"chứ không"` (`"thù luôn chứ không ghét nữa"` → cộng thêm điểm cực trị thay vì đảo dấu). Quy đổi tổng điểm về xác suất `{POS, NEG, NEU}` qua hàm `softmax`/`tanh`. |
| `_build_tfidf_vectorizer()` | Hàm | Tạo `FeatureUnion` gồm 2 nhánh TF-IDF song song: mức từ (`word`, `1–2 n-gram`, `max_features=18000`, `sublinear_tf=True`) và mức ký tự trong biên từ (`char_wb`, `3–5 n-gram`, `max_features=18000`). |
| `_load_vi_corpora() / _load_en_corpora()` | Hàm tải dữ liệu | Tải trực tiếp file CSV/TXT qua `huggingface_hub` và GitHub raw (`h-i-e-u/vietnamese-SA-dataset` + `ura-hcmut/UIT-VSMEC` cho VI; `cardiffnlp/tweeteval` cho EN), tránh hoàn toàn thư viện `datasets`/`pyarrow` bị chặn DLL trên Windows. |
| `_LanguageMLBundle` | Lớp per-language | Quản lý riêng cho từng ngôn ngữ (`vi` hoặc `en`): huấn luyện 3 bộ phân loại cực tính (`MultinomialNB`, `LogisticRegression`, `CalibratedClassifierCV(LinearSVC, cv=3)`) và 1 bộ phân loại 7 cảm xúc (`LogisticRegression`), tính Accuracy/Macro-F1 lưu vào `benchmark_metrics.json`, và cache model xuống đĩa dưới dạng `.joblib`. |
| `ClassicalMLSuite` | Lớp tổng hợp | Cung cấp giao diện thống nhất `ensure_loaded(lang)`, `predict_polarity_all(texts, languages)`, `predict_emotion(texts, languages)` tự định tuyến từng câu về đúng `_LanguageMLBundle("vi")` hoặc `_LanguageMLBundle("en")`. |

### 3.6. `src/models/sentiment_engine.py` (684 dòng) — Giai đoạn 3B: Học sâu, Hybrid Ensemble & Thang điểm kép

| Thành phần | Chi tiết xử lý |
| :--- | :--- |
| `AnalyzedComment` (`dataclass`, 35 trường) | Lưu toàn bộ metadata, kết quả tiền xử lý, xác suất/nhãn/điểm số của Hybrid (`polarity_*`, `score_5`, `score_10`, `emotion_*`), nhãn dự đoán của 7 mô hình riêng lẻ (`pred_nb` .. `emotion_ml_label`), và từ điển `model_probs`. |
| `compute_dual_scale(...)` | Quy đổi xác suất Hybrid thành 3 thang điểm nhất quán tuyệt đối với `polarity_label`:<br>1. Tính độ hích cảm xúc `nudge`: `-0.08 * emo_conf` (nếu `Anger`/`Disgust` ≥ 0.5), `-0.04 * emo_conf` (nếu `Sadness`/`Fear` ≥ 0.5), `+0.06 * emo_conf` (nếu `Enjoyment` ≥ 0.5).<br>2. `raw = p_pos - p_neg + nudge`.<br>3. Khóa vùng theo nhãn:<br>   - `NEG`: kẹp `polarity_score` trong `[-1.0, -0.21]`; `score_5 = 1` nếu `<= -0.60`, ngược lại `2`.<br>   - `POS`: kẹp `polarity_score` trong `[+0.21, +1.0]`; `score_5 = 5` nếu `>= +0.60`, ngược lại `4`.<br>   - `NEU`: kẹp `raw * (1 - 0.5 * p_neu)` trong `[-0.19, +0.19]`; khóa cứng `score_5 = 3`.<br>4. `score_10 = round((polarity_score + 1.0) * 5.0, 2)`. |
| `_HFSequenceClassifier` | Wrapper tải lười mô hình `AutoModelForSequenceClassification` + `AutoTokenizer` trên `CPU` (`batch_size=32`, `max_length=256`, suy luận dưới `torch.inference_mode()`), áp dụng Softmax và chuẩn hóa nhãn qua `_norm_vi_polarity_label`, `_norm_en_polarity_label`, `_norm_emotion_label`. |
| `CommentAnalyzer._apply_emotion_cues(...)` | Cộng thêm xác suất `0.15–0.30` vào đúng cung bậc cảm xúc khi câu chứa cụm từ chỉ báo mạnh (`_FEAR_CUES` như `"scared"`, `"sợ quá"`; `_ANGER_CUES`; `_DISGUST_CUES`; `_SADNESS_CUES`; `_JOY_CUES` như `"peak"`, `"fire"`) rồi chuẩn hóa tổng = 1. |
| Trọng số `Hybrid Ensemble` | - **Nhánh Tiếng Việt (`lang == "vi"`):** `0.40 * PhoBERT + 0.20 * ViSoBERT + 0.25 * Linear_SVM_VI + 0.15 * Lexicon_VI`.<br>- **Nhánh Tiếng Anh (`lang == "en"`):** `0.60 * Twitter_RoBERTa + 0.25 * Linear_SVM_EN + 0.15 * Lexicon_EN`. |
| Đồng bộ `Emotion–Polarity` & `needs_audit` | Nếu `emotion_label ∈ {Anger, Disgust, Fear}` (`conf >= 0.55`) VÀ `Lexicon(NEG) >= 0.50` → cộng `+0.25` vào `NEG`, nhân `0.65` vào `POS` rồi chuẩn hóa lại.<br>Gắn `needs_audit = True` nếu `(ViSoBERT != PhoBERT)` hoặc `(polarity_confidence < 0.65)` hoặc `(has_sarcasm_signal == True)`. |
| `CommentAnalyzer.summarize(...)` | Tổng hợp thống kê toàn bài (`language_counts`, `polarity_counts/percentages`, `emotion_counts/percentages`, `score_5_counts`, điểm trung bình, `overall_tendency` — nếu cả POS và NEG đều ≥ 35% thì kết luận `"Tranh cãi (Trái chiều)"`, và `top_examples` xếp hạng theo `likes` giảm dần). Nếu `deduplicate_stats=True`, các bình luận `is_duplicate=True` bị loại khỏi phép tính thống kê tỷ lệ. |

### 3.7. `src/models/gemma_auditor.py` (653 dòng) — Giai đoạn 4: AI Đầu ra (Thẩm định & Đối chiếu)

- **`GemmaAuditItem` (`dataclass`, 10 trường):** Chứa kết quả thẩm định độc lập của Gemma cho từng bình luận (`comment_index`, `gemma_audited`, `gemma_polarity`, `gemma_emotion`, `gemma_emotion_vi`, `gemma_score_5`, `gemma_is_sarcasm`, `gemma_agrees_polarity`, `gemma_agrees_emotion`, `gemma_reason`).
- **`_format_video_context(media_result)`:** Đóng gói 8 trường ngữ cảnh video từ Giai đoạn 1 thành văn bản ngắn gọn để đưa vào đầu mỗi batch gửi Gemma chấm bình luận.
- **`GemmaAuditor.audit_comments(...)`:** Chia 100% bình luận hợp lệ (hoặc giới hạn `audit_limit` nếu đặt) thành các lô **`chunk_size = 8` bình luận**, gọi `/api/chat` (`temperature=0.1`, ép `BATCH_AUDIT_SCHEMA`). Mỗi câu sau khi nhận JSON đều qua hàm `_reconcile_polarity_and_score()` để bảo đảm `POS ∈ {4, 5}`, `NEU = 3`, `NEG ∈ {1, 2}`.
- **`build_model_comparison(analyzed, audits, timings)`:** Tính phân bố POS/NEU/NEG (%) cho 7 mô hình/nhánh cực tính (`NB`, `LR`, `SVM`, `Lexicon`, `ViSoBERT`, `PhoBERT/RoBERTa`, `Hybrid Ensemble`) + hàng chuẩn `Gemma 4 E2B`, và 2 mô hình cảm xúc (`Emotion LR`, `PhoBERT-Emotion/DistilRoBERTa`). Nếu có dữ liệu thẩm định Gemma, dùng `scikit-learn` tính **4 thước đo đối chiếu**: `Agreement (%)`, `Macro-F1`, `Cohen's Kappa` và `score_5 MAE` (riêng cho Hybrid).
- **`GemmaAuditor.generate_overall_verdict(...)`:** Gửi bảng tổng hợp đối chiếu để Gemma 4 E2B sinh JSON theo `VERDICT_SCHEMA` gồm `accuracy_verdict`, `best_model`, danh sách `strengths`, `weaknesses` và đoạn tóm tắt `summary_vi`.

### 3.8. `src/visualization.py` (192 dòng) & `src/pipeline.py` (573 dòng)

- **`plot_sentiment_dashboard(report, save_path)`:** Vẽ lưới 2×2 biểu đồ kích thước `(15, 10)` inch:
  + Ô `[0, 0]`: Nếu có Gemma Audit → vẽ cột đôi **Hybrid System vs Gemma 4 E2B**; nếu không → vẽ cột đơn **POS / NEU / NEG**.
  + Ô `[0, 1]`: Thanh ngang sắp xếp giảm dần **7 cung bậc cảm xúc**.
  + Ô `[1, 0]`: Biểu đồ cột **Thang điểm 1–5** kèm tiêu đề hiển thị điểm TB thang 5 và thang 10.
  + Ô `[1, 1]`: Nếu có Gemma Audit → thanh ngang đôi **Agreement (%) & Macro-F1 (%) vs Gemma 4**; nếu không → thanh chồng 100% phân bố `POS / NEU / NEG` của 7 mô hình.
- **`SentimentPipeline` (`src/pipeline.py`):** Điều phối khép kín từ `_infer_run_name()` → Bước 0 → Giai đoạn 1 → Xả VRAM → Giai đoạn 2 → Giai đoạn 3 → Giai đoạn 4 → `_export_outputs()` xuất 4 tệp JSON/CSV + `plot_sentiment_dashboard()` xuất `dashboard.png`. Đồng thời cung cấp `main()` chạy bằng dòng lệnh CLI.

---

## 4. Từ điển dữ liệu đầu vào & mô hình lưu đệm (`data/` và `models/`)

### 4.1. `data/<post_id>/metadata.json` — Giải thích đủ 13 thuộc tính

| STT | Tên thuộc tính | Kiểu | Ý nghĩa & Nguồn tạo |
| :--- | :--- | :--- | :--- |
| 1 | `url` | `str` | Đường dẫn TikTok gốc người dùng truyền vào. |
| 2 | `resolved_url` | `str` | Đường dẫn đầy đủ sau khi giải mã short link (`vt.tiktok.com` → `www.tiktok.com/@user/video/...`). |
| 3 | `post_id` | `str` | Mã định danh duy nhất của bài đăng TikTok (ví dụ `"7709458511838612744"`), dùng làm tên thư mục trong cả `data/` và `outputs/`. |
| 4 | `media_type` | `str` | Loại nội dung: `"video"` (bài đăng video) hoặc `"images"` (bài đăng trượt ảnh Photo Mode). |
| 5 | `media_paths` | `list[str]` | Danh sách đường dẫn tới tệp video (`video.mp4`) hoặc các ảnh đã tải về. |
| 6 | `caption` | `str` | Nội dung mô tả bài đăng lấy từ trường `description` của `yt-dlp` (hoặc DOM trang web ở Photo Mode). |
| 7 | `hashtags` | `list[str]` | Danh sách các hashtag `#...` trích xuất từ `caption` qua `extract_hashtags()`. |
| 8 | `uploader` | `str` | Tên định danh (`handle` / `uploader_id`) của chủ kênh đăng video (ví dụ `"ducreaction"`). |
| 9 | `title` | `str` | Tiêu đề bài đăng trên TikTok (do `yt-dlp` trích xuất). |
| 10 | `duration` | `float` \| `null` | Thời lượng video tính bằng giây (`245.0` giây; `null` nếu là bài đăng ảnh). |
| 11 | `like_count` | `int` | Tổng số lượt tim (likes) của bài đăng tại thời điểm thu thập. |
| 12 | `comment_count` | `int` | Tổng số lượng bình luận hiển thị công khai của bài đăng. |
| 13 | `metadata_path` | `str` | Đường dẫn lưu chính tệp `metadata.json` này. |

### 4.2. `data/<post_id>/raw_comments.csv` — Giải thích đủ 8 cột

| STT | Tên cột | Kiểu | Ví dụ thực tế | Ý nghĩa |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `comment` | `str` | `"ê làm sao mà D có được cái giọng..."` | Nội dung nguyên văn của bình luận thu thập từ TikTok (chưa qua lọc hay chuẩn hóa). |
| 2 | `comment_id` | `str` | `"7709509346682979093"` | Mã định danh duy nhất của bình luận trên hệ thống TikTok (`cid`), dùng để loại trùng khi cuộn trang. |
| 3 | `username` | `str` | `"flaycnn"` | Tên tài khoản (`unique_id`) của người bình luận. |
| 4 | `nickname` | `str` | `"mây trời lơ ngơ ☁️"` | Tên hiển thị của người bình luận. |
| 5 | `likes` | `int` | `3230` | Số lượt thích (tim) của bình luận, dùng để sắp xếp các bình luận tiêu biểu (`top_examples`). |
| 6 | `reply_count` | `int` | `16` | Số lượng phản hồi con dưới bình luận này. |
| 7 | `created_at` | `str` | `"2026-11-18 20:54:06"` | Thời điểm đăng bình luận đã quy đổi từ Unix timestamp sang định dạng `YYYY-MM-DD HH:MM:SS`. |
| 8 | `post_url` | `str` | `"https://www.tiktok.com/@ducreaction/..."` | Đường dẫn bài đăng TikTok nơi thu thập bình luận. |

### 4.3. Kho đệm mô hình Học máy cổ điển (`models/classical_ml/{vi,en}/`)

Mỗi thư mục ngôn ngữ (`vi/` và `en/`) chứa 3 tệp được sinh tự động ở lần chạy đầu tiên và tái sử dụng ngay ở các lần chạy sau:
- **`polarity_models.joblib`**: Lưu từ điển gồm bộ vector hóa TF-IDF hỗn hợp (`FeatureUnion` word + char) cùng 3 mô hình đã khớp (`tfidf_nb`, `tfidf_lr`, `tfidf_svm`).
- **`emotion_ml_model.joblib`**: Lưu bộ vector hóa TF-IDF hỗn hợp và mô hình `LogisticRegression` cho bài toán 7 cung bậc cảm xúc.
- **`benchmark_metrics.json`**: Chứa đúng 6 khóa gốc ghi lại kết quả kiểm chứng trên tập test công khai: `polarity_train_size`, `polarity_test_size`, `polarity_models` (có `accuracy`, `macro_f1` cho `tfidf_nb`, `tfidf_lr`, `tfidf_svm`), `emotion_train_size`, `emotion_test_size`, `emotion_lr`.

---

## 5. Từ điển dữ liệu kết quả đầu ra (`outputs/<post_id>/`)

### 5.1. `final_results.csv` — Giải thích đầy đủ toàn bộ 42 cột (kèm chú thích tiếng Việt)

Tệp `final_results.csv` chỉ chứa các **bình luận sạch hợp lệ thuộc Tiếng Việt (`vi`) hoặc Tiếng Anh (`en`)**. Để người đọc mở bằng Excel hiểu ngay ý nghĩa mà không cần tra cứu tài liệu, **tiêu đề mỗi cột đều hiển thị 2 dòng** (dòng trên là tên biến tiếng Anh chuẩn code, dòng dưới là chú thích tiếng Việt trong ngoặc, ví dụ `language` / `(ngôn ngữ)`).

#### Nhóm 1 (Cột 1–6): Thông tin định danh & siêu dữ liệu bình luận (từ Bước 0)

| STT | Tên cột & Chú thích trong CSV | Kiểu | Nguồn & Ý nghĩa chi tiết |
| :--- | :--- | :--- | :--- |
| 1 | `comment_id` `<br>` `(mã bình luận)` | `str` | Mã định danh bình luận lấy từ TikTok (`cid`), giữ nguyên từ `raw_comments.csv`. |
| 2 | `username` `<br>` `(tên tài khoản)` | `str` | Tên đăng nhập (`@handle`) của người viết bình luận. |
| 3 | `nickname` `<br>` `(tên hiển thị)` | `str` | Tên hiển thị hồ sơ của người viết bình luận. |
| 4 | `likes` `<br>` `(lượt thích)` | `int` | Số lượt tim của bình luận; phản ánh mức độ đồng tình của cộng đồng với bình luận đó. |
| 5 | `reply_count` `<br>` `(số phản hồi)` | `int` | Số lượng câu trả lời con dưới bình luận gốc. |
| 6 | `created_at` `<br>` `(thời gian đăng)` | `str` | Thời gian đăng bình luận (`YYYY-MM-DD HH:MM:SS`). |

#### Nhóm 2 (Cột 7–13): Tiền xử lý ngôn ngữ tự nhiên & Cờ phân tích (từ Giai đoạn 2 & 3)

| STT | Tên cột & Chú thích trong CSV | Kiểu | Miền giá trị / Ví dụ | Ý nghĩa chi tiết |
| :--- | :--- | :--- | :--- | :--- |
| 7 | `language` `<br>` `(ngôn ngữ)` | `str` | `"vi"` hoặc `"en"` | Ngôn ngữ được hàm `detect_language()` xác định để định tuyến qua đúng bộ mô hình ML & Deep Learning của ngôn ngữ đó. |
| 8 | `original_text` `<br>` `(nội dung gốc)` | `str` | Nguyên văn tiếng Việt/Anh | Nội dung thô ban đầu người dùng gõ trên TikTok trước khi chuẩn hóa. |
| 9 | `cleaned_text` `<br>` `(nội dung đã làm sạch)` | `str` | Văn bản đã chuẩn hóa | Văn bản sau `normalize_tiktok_emojis()` + `clean_comment_text()`: đã đổi mã `[Cười...]` thành Emoji và đổi teencode an toàn (`mn` → `mọi người`). Là đầu vào cho ViSoBERT, Twitter-RoBERTa, DistilRoBERTa và Gemma 4 E2B. |
| 10 | `segmented_text` `<br>` `(văn bản đã tách từ)` | `str` | Có dấu `_` nối từ ghép (`vi`) | Văn bản sau `_safe_word_tokenize()`: với `vi` do `underthesea` tách từ (`phong_kiến`, `cầu_hôn`), dùng làm đầu vào bắt buộc cho **PhoBERT** và bộ **ML cổ điển**. |
| 11 | `is_duplicate` `<br>` `(bình luận trùng lặp)` | `bool` | `True` / `False` | Đánh dấu `True` nếu nội dung trùng hệt một bình luận hợp lệ đã xuất hiện trước đó (phục vụ tùy chọn `DEDUPLICATE_STATS`). |
| 12 | `has_sarcasm_signal` `<br>` `(dấu hiệu mỉa mai)` | `bool` | `True` / `False` | Cờ cảnh báo sớm ở tầng NLP (chưa kết luận chắc chắn): `True` khi có từ khen đi kèm emoji cười cà khịa `😂`/`🤣` hoặc từ khóa mỉa mai. |
| 13 | `needs_audit` `<br>` `(cần thẩm định kỹ)` | `bool` | `True` / `False` | Cờ gợi ý các bình luận "nhạy cảm cần kiểm tra kỹ": `True` nếu ViSoBERT và PhoBERT bất đồng nhãn, hoặc `polarity_confidence < 0.65`, hoặc `has_sarcasm_signal == True`. |

#### Nhóm 3 (Cột 14–23): Kết quả Hybrid Ensemble & Hệ thang điểm kép (từ Giai đoạn 3)

| STT | Tên cột & Chú thích trong CSV | Kiểu | Miền giá trị | Ý nghĩa chi tiết |
| :--- | :--- | :--- | :--- | :--- |
| 14 | `polarity_label` `<br>` `(nhãn cực tính)` | `str` | `"POS"`, `"NEU"`, `"NEG"` | Nhãn cực tính cuối cùng của hệ thống chính (**Hybrid Ensemble**) chọn theo xác suất lớn nhất sau bước đồng bộ Emotion–Polarity. |
| 15 | `polarity_vi` `<br>` `(cực tính tiếng việt)` | `str` | `"Tích cực"`, `"Trung tính"`, `"Tiêu cực"` | Nhãn hiển thị tiếng Việt tương ứng với `polarity_label` (`LABEL_VI_MAP`). |
| 16 | `polarity_confidence` `<br>` `(độ tự tin cực tính)` | `float` | `0.0000` .. `1.0000` | Mức độ tự tin của hệ thống Hybrid đối với nhãn `polarity_label` đã chọn (`max(score_POS, score_NEG, score_NEU)`). |
| 17 | `polarity_score` `<br>` `(điểm cực tính -1 đến +1)` | `float` | `-1.0000` .. `+1.0000` | Điểm sắc thái liên tục tính bởi `compute_dual_scale()`: luôn khóa `<= -0.21` nếu `NEG`, trong `[-0.19, +0.19]` nếu `NEU`, và `>= +0.21` nếu `POS`. |
| 18 | `score_5` `<br>` `(điểm thang 1-5)` | `int` | `1`, `2`, `3`, `4`, `5` | Điểm thang 5 bậc nhất quán 100% với `polarity_label`: `1` (≤ -0.60), `2` (NEG còn lại), `3` (mọi NEU), `4` (POS còn lại), `5` (≥ +0.60). |
| 19 | `score_10` `<br>` `(điểm thang 0-10)` | `float` | `0.00` .. `10.00` | Quy đổi tuyến tính từ `polarity_score` sang thang điểm 10: `(polarity_score + 1.0) × 5.0`. |
| 20 | `rating_level_vi` `<br>` `(mức độ đánh giá)` | `str` | `"Rất tiêu cực"` .. `"Rất tích cực"` | Diễn giải tiếng Việt tương ứng với 5 mức `score_5` (`RATING_5_VI_MAP`). |
| 21 | `score_POS` `<br>` `(xác suất tích cực)` | `float` | `0.0000` .. `1.0000` | Xác suất tổng hợp của lớp **Tích cực (POS)** từ Hybrid Ensemble (tổng 3 cột 21–23 bằng `1.0`). |
| 22 | `score_NEG` `<br>` `(xác suất tiêu cực)` | `float` | `0.0000` .. `1.0000` | Xác suất tổng hợp của lớp **Tiêu cực (NEG)** từ Hybrid Ensemble. |
| 23 | `score_NEU` `<br>` `(xác suất trung tính)` | `float` | `0.0000` .. `1.0000` | Xác suất tổng hợp của lớp **Trung tính (NEU)** từ Hybrid Ensemble. |

#### Nhóm 4 (Cột 24–26): Kết quả phân tích 7 cung bậc cảm xúc (từ Giai đoạn 3)

| STT | Tên cột & Chú thích trong CSV | Kiểu | Miền giá trị | Ý nghĩa chi tiết |
| :--- | :--- | :--- | :--- | :--- |
| 24 | `emotion_label` `<br>` `(nhãn cảm xúc)` | `str` | `Anger`, `Sadness`, `Fear`, `Enjoyment`, `Disgust`, `Surprise`, `Other` | Nhãn cảm xúc chi tiết chuẩn UIT-VSMEC (do `visolex/phobert-emotion` với `vi`, hoặc ánh xạ từ `j-hartmann/emotion-english-distilroberta-base` với `en`, kết hợp `_apply_emotion_cues`). |
| 25 | `emotion_vi` `<br>` `(cảm xúc 7 cung bậc)` | `str` | `"Tức giận"`, `"Buồn bã"`, ... | Tên tiếng Việt dễ đọc tương ứng của `emotion_label`. |
| 26 | `emotion_confidence` `<br>` `(độ tự tin cảm xúc)` | `float` | `0.0000` .. `1.0000` | Xác suất tự tin của cung bậc cảm xúc chiếm ưu thế nhất. |

#### Nhóm 5 (Cột 27–33): Dự đoán từng mô hình riêng lẻ để so sánh đối chiếu (từ Giai đoạn 3)

| STT | Tên cột & Chú thích trong CSV | Kiểu | Miền giá trị | Ý nghĩa chi tiết |
| :--- | :--- | :--- | :--- | :--- |
| 27 | `pred_nb` `<br>` `(dự đoán naive bayes)` | `str` | `POS`, `NEU`, `NEG` | Nhãn dự đoán riêng của mô hình **TF-IDF + Multinomial Naive Bayes** (đúng nhánh `vi` hoặc `en`). |
| 28 | `pred_lr` `<br>` `(dự đoán logistic regression)` | `str` | `POS`, `NEU`, `NEG` | Nhãn dự đoán riêng của mô hình **TF-IDF + Logistic Regression**. |
| 29 | `pred_svm` `<br>` `(dự đoán linear svm)` | `str` | `POS`, `NEU`, `NEG` | Nhãn dự đoán riêng của mô hình **TF-IDF + Linear SVM (Calibrated)** — mô hình ML tham gia đóng góp 25% vào Hybrid. |
| 30 | `pred_lexicon` `<br>` `(dự đoán từ điển luật)` | `str` | `POS`, `NEU`, `NEG` | Nhãn dự đoán riêng của bộ luật & từ điển **BilingualLexiconScorer** (đóng góp 15% vào Hybrid). |
| 31 | `pred_visobert` `<br>` `(dự đoán visobert)` | `str` | `POS`, `NEU`, `NEG` hoặc `""` | Nhãn dự đoán của **5CD-AI/Vietnamese-Sentiment-visobert**; chỉ có giá trị với comment `vi`, tự động để rỗng (`""`) nếu là comment `en`. |
| 32 | `pred_dl_primary` `<br>` `(dự đoán học sâu chính)` | `str` | `POS`, `NEU`, `NEG` | Nhãn dự đoán của mô hình Học sâu chủ lực đúng ngôn ngữ: **PhoBERT** (khi `language="vi"`) hoặc **Twitter-RoBERTa** (khi `language="en"`). |
| 33 | `emotion_ml_label` `<br>` `(cảm xúc ml cổ điển)` | `str` | 1 trong 7 nhãn cảm xúc | Nhãn cảm xúc dự đoán bởi mô hình **TF-IDF + Logistic Regression (Emotion)** để đối chiếu với Deep Learning Emotion ở cột 24. |

#### Nhóm 6 (Cột 34–42): Thẩm định thực tế theo ngữ cảnh của Gemma 4 E2B (từ Giai đoạn 4)

| STT | Tên cột & Chú thích trong CSV | Kiểu | Miền giá trị | Ý nghĩa chi tiết |
| :--- | :--- | :--- | :--- | :--- |
| 34 | `gemma_audited` `<br>` `(đã thẩm định gemma)` | `bool` | `True` / `False` | Xác nhận bình luận này đã được Gemma 4 E2B chấm thực tế ở Giai đoạn 4 hay chưa (sẽ là `False` nếu vượt `AUDIT_LIMIT` hoặc tắt `RUN_AI_AUDIT`). |
| 35 | `gemma_polarity` `<br>` `(cực tính gemma)` | `str` | `POS`, `NEU`, `NEG` hoặc `""` | Nhãn cực tính do **Gemma 4 E2B** đánh giá độc lập dựa trên cả bình luận lẫn ngữ cảnh video Giai đoạn 1. |
| 36 | `gemma_emotion` `<br>` `(nhãn cảm xúc gemma)` | `str` | 1 trong 7 nhãn tiếng Anh hoặc `""` | Cung bậc cảm xúc do Gemma 4 E2B xác định. |
| 37 | `gemma_emotion_vi` `<br>` `(cảm xúc gemma)` | `str` | Tên cảm xúc tiếng Việt hoặc `""` | Tên tiếng Việt của `gemma_emotion`. |
| 38 | `gemma_score_5` `<br>` `(điểm 1-5 gemma)` | `int` \| `""` | `1..5` (hoặc rỗng) | Điểm thang 5 do Gemma 4 E2B chấm (đã qua `_reconcile_polarity_and_score` để `NEG∈{1,2}`, `NEU=3`, `POS∈{4,5}`). |
| 39 | `gemma_is_sarcasm` `<br>` `(nói mỉa theo gemma)` | `bool` \| `""` | `True` / `False` | Kết luận thực tế của Gemma về việc bình luận này có mang hàm ý **nói mỉa / châm biếm** hay không. |
| 40 | `gemma_agrees_polarity` `<br>` `(khớp cực tính với gemma)` | `bool` \| `""` | `True` / `False` | `True` khi `gemma_polarity == polarity_label` (Hybrid hệ thống chính đoán đúng khớp với Gemma). |
| 41 | `gemma_agrees_emotion` `<br>` `(khớp cảm xúc với gemma)` | `bool` \| `""` | `True` / `False` | `True` khi `gemma_emotion == emotion_label` (Deep Learning cảm xúc khớp với Gemma). |
| 42 | `gemma_reason` `<br>` `(lý do gemma giải thích)` | `str` | Chuỗi tiếng Việt ngắn | Lời giải thích ngắn gọn lý do vì sao Gemma 4 E2B chấm nhãn cực tính và cảm xúc như trên (giúp người đọc kiểm chứng minh bạch). |

---

### 5.2. `cleaned_final_results.csv` — Bản sao tóm gọn 15 cột quan trọng (có chú thích tiếng Việt)

Tệp `cleaned_final_results.csv` là bản sao rút gọn trực tiếp từ `final_results.csv`, giữ lại **15 cột quan trọng nhất** (tự động rút còn **10 cột** cốt lõi nếu lượt chạy tắt `RUN_AI_AUDIT = False`) và **sắp xếp sẵn theo `likes` giảm dần** để các bình luận có nhiều lượt tim nhất nổi lên đầu bảng. Tiêu đề mỗi cột cũng chứa tên tiếng Việt nằm ngay dưới tên tiếng Anh.

| STT | Tên cột & Chú thích trong CSV | Nhóm thông tin | Vì sao nên giữ lại & Ý nghĩa |
| :---: | :--- | :--- | :--- |
| 1 | `username` `<br>` `(tên tài khoản)` | Người bình luận | Biết ai bình luận (gọn hơn giữ đồng thời `comment_id`, `nickname` và `username`). |
| 2 | `likes` `<br>` `(lượt thích)` | Tương tác | Đánh giá mức độ đồng tình của cộng đồng; làm khóa sắp xếp các bình luận nổi bật lên đầu. |
| 3 | `created_at` `<br>` `(thời gian đăng)` | Thời gian | Theo dõi diễn biến cảm xúc theo thời điểm đăng bình luận. |
| 4 | `language` `<br>` `(ngôn ngữ)` | Ngôn ngữ | Phân biệt ngay bình luận Tiếng Việt (`vi`) hay Tiếng Anh (`en`). |
| 5 | `original_text` `<br>` `(nội dung gốc)` | Nội dung | Câu bình luận gốc nguyên văn (dễ đọc nhất khi kiểm tra bằng mắt thường). |
| 6 | `polarity_vi` `<br>` `(cực tính hệ thống)` | Kết quả hệ thống | Nhãn cực tính chính dễ hiểu (`Tích cực` / `Trung tính` / `Tiêu cực`). |
| 7 | `score_5` `<br>` `(điểm thang 1-5)` | Kết quả hệ thống | Điểm thang `1–5` sao trực quan nhất. |
| 8 | `score_10` `<br>` `(điểm thang 0-10)` | Kết quả hệ thống | Điểm thang `0–10` chi tiết (phân biệt mức độ nhẹ/mạnh trong cùng bậc sao). |
| 9 | `emotion_vi` `<br>` `(cảm xúc 7 cung bậc)` | Kết quả hệ thống | 1 trong 7 cung bậc cảm xúc chi tiết (`Vui vẻ / Hài lòng`, `Chán ghét / Khó chịu`, `Buồn bã`...). |
| 10 | `polarity_confidence` `<br>` `(độ tự tin cực tính)` | Độ tin cậy | Biết hệ thống tự tin bao nhiêu (`0.00` – `1.00`) với dự đoán cực tính của mình. |
| 11 | `gemma_polarity_vi` `<br>` `(cực tính gemma tiếng việt)` | Thẩm định AI | Quy đổi nhãn của Gemma 4 E2B sang tiếng Việt (`Tích cực` / `Trung tính` / `Tiêu cực`) để đối chiếu ngay cạnh cột 6. *(Tự ẩn nếu tắt Audit)* |
| 12 | `gemma_score_5` `<br>` `(điểm 1-5 gemma)` | Thẩm định AI | Điểm `1–5` do Gemma 4 E2B chấm để so ngay với cột 7. *(Tự ẩn nếu tắt Audit)* |
| 13 | `gemma_is_sarcasm` `<br>` `(nói mỉa theo gemma)` | Phát hiện mỉa mai | Cho biết bình luận có phải nói mỉa / châm biếm hay không (`True` / `False`). *(Tự ẩn nếu tắt Audit)* |
| 14 | `gemma_agrees_polarity` `<br>` `(khớp cực tính với gemma)` | Đối chiếu đúng/sai | Trả lời trực tiếp *"Hệ thống chính có đoán khớp với Gemma không?"*. *(Tự ẩn nếu tắt Audit)* |
| 15 | `gemma_reason` `<br>` `(lý do gemma giải thích)` | Giải thích AI | Lý do ngắn gọn bằng tiếng Việt vì sao Gemma đánh giá như vậy. *(Tự ẩn nếu tắt Audit)* |

---

### 5.3. `filtered_noise.csv` — Giải thích đủ 8 cột & 7 mã lọc nhiễu

Tệp này lưu toàn bộ bình luận bị loại bỏ khỏi phân tích cảm xúc để người dùng đối chiếu không lo hệ thống "xóa nhầm dữ liệu mà không báo".

| STT | Tên cột | Kiểu | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| 1 | `comment_id` | `str` | Mã định danh bình luận trên TikTok. |
| 2 | `username` | `str` | Tài khoản người bình luận. |
| 3 | `nickname` | `str` | Tên hiển thị của người bình luận. |
| 4 | `likes` | `int` | Số lượt thích của bình luận bị lọc. |
| 5 | `language` | `str` | Kết quả nhận diện ngôn ngữ từ Giai đoạn 2: `"vi"`, `"en"`, hoặc `"other"` (ví dụ câu tiếng Nga, Tây Ban Nha, Bồ Đào Nha). |
| 6 | `original_text` | `str` | Nội dung gốc nguyên văn của bình luận bị loại. |
| 7 | `cleaned_text` | `str` | Nội dung sau bước làm sạch khoảng trắng / chuẩn hóa emoji chữ. |
| 8 | `filter_reason` | `str` | Mã lý do lọc (1 trong 7 giá trị chuẩn dưới đây). |

**Ý nghĩa 7 giá trị của cột `filter_reason`:**
1. `FILTERED_EMPTY`: Chuỗi rỗng hoặc chỉ có khoảng trắng.
2. `FILTERED_UNSUPPORTED_LANG`: Ngôn ngữ khác ngoài Tiếng Việt và Tiếng Anh (`language == "other"`).
3. `FILTERED_URL_OR_SYMBOL`: Chỉ chứa đường link URL, con số rời rạc, dấu chấm câu hoặc toàn bộ là icon/emoji (ví dụ `"😂😂😂"`, `"🔥🔥"`).
4. `FILTERED_TAG_FRIEND`: Bình luận chỉ gắn thẻ tài khoản `@username` hoặc gọi tên bạn bè đi kèm 1–2 từ kêu gọi ngắn (ví dụ `"@Băng Tâm 😂"`, `"@chloe @Leah"`).
5. `FILTERED_NAME_ONLY`: Chỉ gồm tên riêng người Việt (từ 2 từ trở lên, phát hiện qua `underthesea.ner`/`pos_tag`) mà không chứa từ bày tỏ cảm xúc.
6. `FILTERED_SPAM_AD`: Bình luận chứa từ khóa quảng cáo, bán hàng, tuyển sỉ, mời nhắn tin Zalo/IB.
7. `FILTERED_MEANINGLESS`: Các từ rác đơn lẻ không có nghĩa đánh giá như `"fyp"`, `"hóng"`, `"xin in4"`, `"chấm"`, `"."`.

---

### 5.4. `model_comparison.csv` — Giải thích đủ 15 cột

Tệp này tổng hợp bảng so sánh hiệu năng toàn diện giữa ML cổ điển, Luật từ điển, Deep Learning, Hybrid Ensemble và Gemma 4 E2B (có tối đa 10 dòng: 8 dòng cho bài toán `Polarity (3 lớp)` và 2 dòng cho bài toán `Emotion (7 cung bậc)`).

| STT | Tên cột | Kiểu | Ý nghĩa & Giải thích các trường hợp ô trống |
| :--- | :--- | :--- | :--- |
| 1 | `model` | `str` | Tên mô hình hoặc tổ hợp (ví dụ `TF-IDF + Linear SVM`, `Hybrid Ensemble (Hệ thống chính)`, `Gemma 4 E2B (Thẩm định ngữ cảnh)`). |
| 2 | `family` | `str` | Họ phương pháp: `Classical ML`, `Rule-based NLP`, `Deep Learning`, `Hybrid ML + DL`, hoặc `Multimodal LLM Reference`. |
| 3 | `total_comments` | `float` | Số lượng bình luận mà mô hình này dự đoán trên tập hợp lệ (ví dụ ViSoBERT chỉ đếm riêng các bình luận `vi`, còn Hybrid đếm tất cả `vi + en`). Để trống ở 2 hàng bài toán Emotion. |
| 4 | `pos_count` | `float` | Số bình luận được mô hình dự đoán là Tích cực (`POS`). |
| 5 | `neu_count` | `float` | Số bình luận được mô hình dự đoán là Trung tính (`NEU`). |
| 6 | `neg_count` | `float` | Số bình luận được mô hình dự đoán là Tiêu cực (`NEG`). |
| 7 | `pos_pct` | `float` | Tỷ lệ `%` dự đoán `POS` tính trên `total_comments` của mô hình đó. |
| 8 | `neu_pct` | `float` | Tỷ lệ `%` dự đoán `NEU`. |
| 9 | `neg_pct` | `float` | Tỷ lệ `%` dự đoán `NEG`. |
| 10 | `agreement_vs_gemma_pct` | `float` \| rỗng | Tỷ lệ `%` mô hình đoán trùng nhãn với Gemma 4 E2B (đối với hàng Emotion là tỷ lệ trùng 7 cung bậc cảm xúc). **Rỗng nếu `RUN_AI_AUDIT = False`**. |
| 11 | `macro_f1_vs_gemma` | `float` \| rỗng | Điểm cân bằng `Macro-F1` (0..1) so với nhãn Gemma 4 E2B; giúp phát hiện mô hình bị đoán thiên lệch 1 lớp. |
| 12 | `cohen_kappa_vs_gemma` | `float` \| rỗng | Hệ số đồng thuận Cohen's Kappa (-1..+1) đã trừ yếu tố trùng hợp ngẫu nhiên. Để trống ở 2 hàng Emotion. |
| 13 | `score_5_mae_vs_gemma` | `float` \| rỗng | Sai số tuyệt đối trung bình trên thang 1–5 sao so với `gemma_score_5`. **Chỉ có giá trị ở hàng `Hybrid Ensemble` và `Gemma 4 E2B (0.0)`** vì các mô hình đơn lẻ không xuất điểm 5 bậc. |
| 14 | `inference_time_sec` | `float` \| rỗng | Thời gian chạy thực tế (giây) đo bằng `time.perf_counter()` cho mô hình/nhánh tương ứng (3 mô hình NB/LR/SVM dùng chung tổng thời gian của bước `classical_ml_polarity_sec`; hàng Hybrid là phép cộng trọng số tức thời nên để trống). |
| 15 | `task` | `str` | Phân loại bài toán của hàng đó: `"Polarity (3 lớp)"` hoặc `"Emotion (7 cung bậc)"`. |

---

### 5.5. `report.json` — Giải thích cấu trúc 10 nhánh JSON gốc

| STT | Khóa gốc (`report.json`) | Kiểu | Nội dung lưu trữ |
| :--- | :--- | :--- | :--- |
| 1 | `run_name` | `str` | Mã định danh lượt chạy (`<post_id>`, ví dụ `"7709458511838612744"`). |
| 2 | `tiktok_metadata` | `dict` \| `null` | Toàn bộ 13 trường của `metadata.json` (sẽ là `null` nếu người dùng chạy thẳng từ tệp CSV cục bộ mà không qua `TIKTOK_URL`). |
| 3 | `media_analysis` | `dict` \| `null` | Toàn bộ 15 trường từ Giai đoạn 1 (`transcript`, `chu_de_chinh`, `muc_dich_chinh`, `quan_diem_noi_dung`, `doi_tuong_chinh`, `sac_thai_noi_dung`, `tom_tat_ngan`, `tu_khoa`, `luu_y_danh_gia_comment`...). |
| 4 | `noise_filter_summary` | `dict` | Gồm 5 khóa: `total_raw_comments`, `valid_comments`, `filtered_noise_count`, `duplicate_valid_count`, và `filter_reasons` (đếm số lượng từng mã lỗi). |
| 5 | `keywords_tfidf` | `dict` | Gồm 3 danh sách (`overall`, `positive`, `negative`), mỗi phần tử gồm `keyword` và `tfidf_score` (Top 10 từ khóa sạch nổi bật nhất). |
| 6 | `sentiment_summary` | `dict` | Gồm 16 trường thống kê tổng hợp từ `CommentAnalyzer.summarize()`: phân bố ngôn ngữ, polarity, 7 cảm xúc, thang điểm 1–5, điểm trung bình 3 thang, `overall_tendency`, `classical_ml_benchmark`, `model_timings_sec`, `top_examples`. |
| 7 | `model_comparison` | `dict` | Gồm 2 danh sách: `polarity_models` và `emotion_models` (dữ liệu nguồn để xuất `model_comparison.csv`). |
| 8 | `ai_audit_verdict` | `dict` \| `null` | Tổng kết từ Gemma 4 E2B ở cuối Giai đoạn 4: `accuracy_verdict`, `best_model`, `strengths`, `weaknesses`, `summary_vi`, `audited_count`, `sarcasm_detected_count`, `sarcasm_rate_pct`. |
| 9 | `comments` | `list[dict]` | Danh sách đầy đủ từng bình luận hợp lệ gồm toàn bộ 35 trường của `AnalyzedComment` (kèm từ điển xác suất `model_probs` của từng mô hình) + nhánh con `gemma_audit`. |
| 10 | `filtered_noise` | `list[dict]` | Danh sách các bình luận nhiễu (`language`, `original_text`, `cleaned_text`, `filter_reason`). |

---

### 5.6. `dashboard.png` — Hướng dẫn đọc 4 biểu đồ (lưới 2×2)

Biểu đồ do `plot_sentiment_dashboard()` trong `src/visualization.py` sinh ra có phần tiêu đề trên cùng hiển thị **Chủ đề chính** (do Gemma 4 E2B rút ra ở Giai đoạn 1), kèm số lượng **Bình luận hợp lệ (`VI` / `EN`)** và **Đã lọc nhiễu**:
- **Góc trên – trái (`Sắc thái Polarity`):** Khi có thẩm định Gemma (`RUN_AI_AUDIT=True`), vẽ biểu đồ cột đôi so sánh trực tiếp số lượng `POS / NEU / NEG` giữa **Hybrid System** và **Gemma 4 E2B**. Khi tắt audit (`RUN_AI_AUDIT=False`), vẽ cột đơn phân bố 3 nhãn kèm xu hướng chung.
- **Góc trên – phải (`Phân bố 7 cung bậc cảm xúc`):** Biểu đồ thanh ngang xếp từ cảm xúc xuất hiện nhiều nhất xuống ít nhất (`Vui vẻ / Hài lòng`, `Chán ghét / Khó chịu`, `Buồn bã`, `Ngạc nhiên`, `Tức giận`, `Lo lắng / Sợ hãi`, `Trung tính / Khác`).
- **Góc dưới – trái (`Thang điểm 1–5`):** Đếm số bình luận rơi vào từng mức sao từ `1 (Rất tiêu cực)` đến `5 (Rất tích cực)` kèm giá trị trung bình thang 5 và thang 10 trên tiêu đề phụ.
- **Góc dưới – phải (`So sánh các mô hình`):** Khi có Gemma Audit, vẽ thanh ngang so sánh **Agreement (%)** và **Macro-F1 (%)** của từng mô hình (`NB`, `LR`, `SVM`, `Lexicon`, `ViSoBERT`, `PhoBERT/Twitter-RoBERTa`, `Hybrid Ensemble`) so với Gemma 4 E2B. Khi chưa chạy Audit, tự động chuyển sang biểu đồ thanh chồng 100% tỷ lệ `POS % / NEU % / NEG %` để đối chiếu độ thiên lệch giữa các mô hình.

---

## 6. Bảng tra cứu tham số cấu hình & vận hành (`main.ipynb` & CLI)

| Tham số (`SentimentPipeline.run`) | Cờ CLI (`python -m src.pipeline`) | Mặc định | Tác dụng & Khuyến nghị chỉnh |
| :--- | :--- | :--- | :--- |
| `tiktok_url` | `--url` | `None` | Dán link video/ảnh TikTok (hỗ trợ cả `vt.tiktok.com` lẫn `www.tiktok.com`). |
| `media_input` | `--media` | `None` | Truyền đường dẫn video/ảnh cục bộ nếu không tải từ link TikTok. |
| `comments_source` | `--comments` | `None` | Truyền tệp `.csv`, `.json`, hoặc `.txt` nếu muốn chạy lại trên dữ liệu bình luận đã có sẵn. |
| `enable_audio` | `--no-audio` | `True` | Bật/tắt bước giải mã âm thanh bằng PyAV + Whisper ở Giai đoạn 1. Giữ `True` với video nói/review; có thể tắt nếu là bài ảnh hoặc clip chỉ có nhạc nền. |
| `run_ai_audit` | `--skip-audit` | `True` | Bật/tắt Giai đoạn 4 (Gemma 4 E2B thẩm định đầu ra). Bật khi cần kiểm chứng thực tế & chấm điểm đối chiếu; tắt nếu chỉ cần thống kê nhanh trong ~10 giây. |
| `audit_limit` | `--audit-limit` | `None` | `None` = chấm **100%** bình luận hợp lệ; đặt số nguyên (ví dụ `80`) nếu muốn giới hạn thời gian chạy trên video có quá nhiều bình luận. |
| `max_comments` | `--max-comments` | `200` | Số lượng bình luận thô tối đa cuộn thu thập trên trình duyệt Edge. |
| `deduplicate_stats` | `--deduplicate-stats` | `False` | Đặt `True` nếu muốn loại bỏ các bình luận copy-paste lặp lại (`is_duplicate=True`) khỏi bảng tỷ lệ `%` tổng hợp. |
| `unload_ollama_after_stage1` | *(mặc định bật)* | `True` | Tự động gửi `keep_alive=0` sau Giai đoạn 1 để giải phóng VRAM 4 GB; giữ nguyên `True`. |
| `data_dir` | `--data-dir` | `"data"` | Thư mục gốc lưu toàn bộ dữ liệu thu thập thô (`data/<post_id>/`). |
| `output_dir` | `--output` | `"outputs"` | Thư mục gốc xuất toàn bộ kết quả xử lý cuối cùng (`outputs/<post_id>/`). |
