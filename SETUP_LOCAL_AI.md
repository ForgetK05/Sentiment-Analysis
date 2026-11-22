# Hướng dẫn Cài đặt & Vận hành Local AI (`Gemma 4 E2B` + `hand_env`)

Tài liệu này hướng dẫn từng bước thiết lập hệ thống **AI cục bộ (Local AI)** chạy hoàn toàn trên máy cá nhân bằng **Docker Ollama** với mô hình đa phương thức **Gemma 4 E2B (`gemma4-e2b-opt`)**, kết hợp môi trường Python **`hand_env`** tối ưu cho card đồ họa giới hạn **4 GB VRAM** (như NVIDIA RTX 3050 Laptop).

> **Lợi ích khi chạy Local AI:** Không tốn phí API đám mây, không gửi video hay dữ liệu bình luận ra bên ngoài, và hỗ trợ hiểu đồng thời cả **hình ảnh khung hình video** lẫn **văn bản tiếng Việt / tiếng Anh**.

---

## 1. Yêu cầu phần cứng & phần mềm tiên quyết

### 1.1. Phần cứng khuyến nghị
- **Hệ điều hành:** Windows 10/11 (64-bit).
- **Card đồ họa (GPU):** NVIDIA GPU hỗ trợ CUDA với tối thiểu **4 GB VRAM** (đã kiểm chứng ổn định trên NVIDIA GeForce RTX 3050 Laptop 4 GB).
- **RAM hệ thống:** Tối thiểu **16 GB RAM** (do Giai đoạn 2–3 chạy `Faster-Whisper` và các mô hình `PhoBERT` / `RoBERTa` trên CPU + RAM).
- **Dung lượng ổ đĩa trống:** Tối thiểu **15 GB** để chứa ảnh Docker Ollama, trọng số `gemma4:e2b`, bộ mô hình HuggingFace và `Faster-Whisper`.

### 1.2. Phần mềm cần cài trước
1. **NVIDIA GPU Driver** mới nhất — kiểm tra trong PowerShell:
   ```powershell
   nvidia-smi
   ```
   *(Phải thấy bảng tên GPU, dung lượng VRAM và phiên bản CUDA).*
2. **Docker Desktop for Windows:** Bật chế độ **WSL 2 based engine** (Settings → General) để container tự nhận GPU NVIDIA qua `NVIDIA Container Toolkit` tích hợp sẵn.
3. **Anaconda hoặc Miniconda** (quản lý môi trường Python 3.12 `hand_env`).
4. **Microsoft Edge** (cài sẵn trên Windows 10/11; được Playwright dùng khi thu thập bình luận TikTok).

---

## 2. Bước 1 — Khởi tạo Container Ollama GPU

Thư mục cấu hình nằm tại `e:\A\Sentiment-Analysis\AI_Local\gemma4-e2b\`:
- `docker-compose.yml`: cấu hình container tên `gemma4-e2b` lắng nghe cổng `11434`.
- `Modelfile`: định nghĩa mô hình tùy biến `gemma4-e2b-opt` dựa trên `gemma4:e2b`.

### 2.1. Kiểm tra đường dẫn ánh xạ ổ đĩa (`volumes`)
Mở `AI_Local/gemma4-e2b/docker-compose.yml` và kiểm tra 2 dòng dưới mục `volumes`:
```yaml
volumes:
  - E:/A/Sentiment-Analysis/AI_Local/gemma4-e2b/ollama_models:/root/.ollama
  - E:/A/Sentiment-Analysis/AI_Local/gemma4-e2b/Modelfile:/Modelfile
```
*(Nếu bạn sao chép dự án sang ổ đĩa hoặc thư mục khác, hãy sửa lại đúng đường dẫn tuyệt đối tương ứng trên máy của bạn).*

### 2.2. Ý nghĩa các biến môi trường tối ưu 4 GB VRAM
- `OLLAMA_FLASH_ATTENTION=1`: Bật Flash Attention giúp tiết kiệm VRAM khi xử lý chuỗi nhiều khung hình ảnh.
- `OLLAMA_KV_CACHE_TYPE=q8_0`: Lượng tử hóa bộ nhớ đệm ngữ cảnh (KV Cache) xuống 8-bit, giảm khoảng **40%** dung lượng VRAM tiêu thụ.
- `OLLAMA_MAX_LOADED_MODELS=1`: Chỉ cho phép giữ 1 mô hình trên GPU tại một thời điểm.
- `OLLAMA_KEEP_ALIVE=10m`: Giữ mô hình trong VRAM tối đa 10 phút khi không có yêu cầu mới (khi chạy Pipeline chính, code Python sẽ tự gọi `keep_alive=0` ở cuối Giai đoạn 1 để xả VRAM ngay).
- `OLLAMA_HOST=0.0.0.0`: Cho phép Python (`http://localhost:11434`) gọi API từ bên ngoài container.

### 2.3. Lệnh khởi chạy và kiểm tra GPU trong Container
Mở PowerShell tại thư mục `AI_Local\gemma4-e2b`:
```powershell
cd E:\A\Sentiment-Analysis\AI_Local\gemma4-e2b
docker compose up -d
```

Kiểm tra container đang chạy và đã nhận GPU NVIDIA:
```powershell
docker ps
docker exec -it gemma4-e2b nvidia-smi
```
*(Nếu `nvidia-smi` bên trong container hiển thị đúng card đồ họa RTX của bạn thì cấu hình GPU đã thành công).*

---

## 3. Bước 2 — Tải `gemma4:e2b` và tạo mô hình `gemma4-e2b-opt`

### 3.1. Giải thích cấu hình `Modelfile`
Tệp `AI_Local/gemma4-e2b/Modelfile` thiết lập các tham số nền tảng:
```dockerfile
FROM gemma4:e2b

# Mở rộng ngữ cảnh từ 4K mặc định lên 16K tokens (đủ chứa tối đa 10 khung hình video + prompt)
PARAMETER num_ctx 16384

# Ép nạp toàn bộ các lớp mạng lên GPU rời
PARAMETER num_gpu 99

# Tham số sinh văn bản chuẩn của Gemma 4
PARAMETER temperature 1.0
PARAMETER top_p 0.95
PARAMETER top_k 64
```
*(Khi chạy Giai đoạn 1 và Giai đoạn 4, code Python sẽ chủ động truyền `temperature=0.15` / `0.10` cùng `JSON Schema` để ép đầu ra đúng định dạng dữ liệu).*

### 3.2. Tải trọng số và build mô hình tùy biến
Chạy lần lượt 3 lệnh sau trong PowerShell:
```powershell
# 1. Tải mô hình gốc gemma4:e2b (lưu vào thư mục ollama_models/)
docker exec -it gemma4-e2b ollama pull gemma4:e2b

# 2. Tạo bản tùy biến tối ưu 16K context mang tên gemma4-e2b-opt
docker exec -it gemma4-e2b ollama create gemma4-e2b-opt -f /Modelfile

# 3. Xác nhận danh sách mô hình đã sẵn sàng
docker exec -it gemma4-e2b ollama list
```
Nếu ở thư mục dự án đã có sẵn thư mục `ollama_models/` (đã tải trước trọng số), lệnh `ollama pull` sẽ kiểm tra rất nhanh và `ollama list` sẽ hiển thị ngay `gemma4-e2b-opt:latest`.

---

## 4. Bước 3 — Kiểm thử Local AI trước khi chạy Pipeline

1. **Kiểm tra API REST phản hồi:**
   ```powershell
   Invoke-RestMethod http://localhost:11434/api/tags | ConvertTo-Json -Depth 4
   ```
2. **Kiểm tra sinh văn bản trực tiếp:**
   ```powershell
   docker exec -it gemma4-e2b ollama run gemma4-e2b-opt "Trả lời ngắn gọn 1 câu bằng tiếng Việt: Sentiment Analysis là gì?"
   ```
3. **Kiểm tra nạp GPU và tỷ lệ chia CPU/GPU:**
   ```powershell
   docker exec -it gemma4-e2b ollama ps
   ```
   *(Kiểm tra cột `PROCESSOR` hiển thị phần lớn hoặc `100% GPU`).*
4. **Kiểm tra khả năng đọc hình ảnh/video đa phương thức:**
   Mở và chạy thử notebook `AI_Local/gemma4-e2b/test.ipynb` với một tệp video `.mp4` bất kỳ.
5. **Xả sạch VRAM thủ công (khi cần nhường GPU cho tác vụ khác):**
   ```powershell
   Invoke-RestMethod -Method Post -Uri http://localhost:11434/api/generate -ContentType "application/json" -Body '{"model":"gemma4-e2b-opt","keep_alive":0}'
   ```

---

## 5. Bước 4 — Thiết lập môi trường Python (`hand_env`)

### 5.1. Tạo môi trường và cài đặt thư viện
Mở **Anaconda Prompt** hoặc **PowerShell** tại gốc dự án:
```powershell
cd E:\A\Sentiment-Analysis
conda create -n hand_env python=3.12 -y
conda activate hand_env
pip install -r requirements.txt
```

### 5.2. Kiểm tra trình duyệt cho bước thu thập TikTok (Playwright)
Hệ thống dùng kênh trình duyệt **Microsoft Edge** có sẵn trên Windows (`channel="msedge"`). Nếu máy chưa từng khởi tạo Playwright, chạy lệnh sau để đảm bảo Playwright nhận đủ thành phần hỗ trợ:
```powershell
playwright install msedge
```

### 5.3. Chuẩn bị ở lần chạy đầu tiên (tải tự động)
Ở lần chạy `main.ipynb` hoặc `python -m src.pipeline` đầu tiên, hệ thống sẽ tự động tải và lưu cache cục bộ các thành phần phụ trợ sau (những lần sau sẽ chạy ngoại tuyến ngay lập tức):
- **Faster-Whisper (`small`, CPU `int8`):** Tự tải để bóc tách lời thoại đa ngôn ngữ từ video.
- **Bộ mô hình ML cổ điển (`models/classical_ml/vi/` & `en/`):** Tự tải tập dữ liệu công khai qua `huggingface_hub` / `requests`, huấn luyện NB/LR/SVM/Emotion LR và lưu cache thành tệp `.joblib`.
- **Bộ mô hình Học sâu Transformer (CPU):** Tải lười theo ngôn ngữ xuất hiện (`PhoBERT`, `ViSoBERT`, `PhoBERT Emotion` cho Tiếng Việt; `Twitter-RoBERTa`, `DistilRoBERTa Emotion` cho Tiếng Anh).

---

## 6. Vận hành hằng ngày & Khắc phục lỗi thường gặp (Troubleshooting)

### 6.1. Lệnh bật / tắt nhanh Local AI hằng ngày
```powershell
# Bật Local AI trước khi mở main.ipynb
cd E:\A\Sentiment-Analysis\AI_Local\gemma4-e2b
docker compose up -d

# Tắt hẳn container sau khi làm việc xong
docker compose down
```

### 6.2. Bảng xử lý 5 lỗi thường gặp

| Hiện tượng / Thông báo lỗi | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| `ConnectionRefusedError` / `Failed to establish a new connection: [WinError 10061]` tới `localhost:11434` | Container `gemma4-e2b` chưa khởi động hoặc Docker Desktop chưa bật. | Mở Docker Desktop, sau đó chạy `docker compose up -d` trong `AI_Local\gemma4-e2b`. |
| `404 Client Error: model 'gemma4-e2b-opt' not found` | Đã bật container nhưng chưa tạo mô hình tùy biến từ `/Modelfile`. | Chạy `docker exec -it gemma4-e2b ollama create gemma4-e2b-opt -f /Modelfile`. |
| Chạy Gemma rất chậm hoặc báo thiếu VRAM / tràn sang CPU | Đang mở game/app nặng chiếm VRAM hoặc `num_ctx 16384` vượt ngưỡng trống hiện tại của GPU 4 GB. | Tắt các ứng dụng tốn GPU (trình duyệt nặng, phần mềm đồ họa); kiểm tra `nvidia-smi`; nếu vẫn thiếu, giảm `PARAMETER num_ctx` trong `Modelfile` xuống `8192` hoặc `12288` rồi chạy lại `ollama create`. |
| Lỗi `ffmpeg is not installed` khi tải video TikTok | Máy Windows chưa cài đặt FFmpeg toàn cục. | Không cần cài thủ công: `src/ingestion/tiktok_collector.py` đã tự kiểm tra `shutil.which("ffmpeg")` và chuyển sang luồng MP4 ghép sẵn (`b[ext=mp4]/best[ext=mp4]/b`), đồng thời dùng **PyAV** tách âm thanh trực tiếp. |
| Chính sách bảo mật Windows chặn DLL `pyarrow._fs` | Windows Application Control chặn file nhị phân của `pyarrow`. | Dự án đã loại bỏ hoàn toàn phụ thuộc `datasets`/`pyarrow` khỏi `classical_ml.py` và tải trực tiếp CSV/TXT qua `huggingface_hub` và `requests`. Không cài lại `pyarrow` nếu không cần thiết. |
