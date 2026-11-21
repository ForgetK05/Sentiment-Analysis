from __future__ import annotations

import base64
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import requests

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

MEDIA_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chu_de_chinh": {
            "type": "string",
            "description": "Chủ đề chính của video hoặc hình ảnh TikTok.",
        },
        "quan_diem_noi_dung": {
            "type": "string",
            "description": "Quan điểm, thông điệp hoặc góc nhìn chính mà nội dung muốn truyền tải.",
        },
        "tom_tat_ngan": {
            "type": "string",
            "description": "Tóm tắt ngắn gọn nội dung từ hình ảnh và âm thanh.",
        },
        "tu_khoa": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Danh sách 3-6 từ khóa trọng tâm liên quan đến chủ đề.",
        },
    },
    "required": [
        "chu_de_chinh",
        "quan_diem_noi_dung",
        "tom_tat_ngan",
        "tu_khoa",
    ],
}


@dataclass
class MediaAnalysisResult:
    """Kết quả phân tích nội dung đa phương tiện (video/ảnh) từ Gemma 4 E2B."""

    media_type: str
    source_paths: list[str]
    num_frames_used: int
    has_audio: bool
    transcript: str
    chu_de_chinh: str
    quan_diem_noi_dung: str
    tom_tat_ngan: str
    tu_khoa: list[str]
    raw_response: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MediaAnalyzer:
    """Giai đoạn 1: Trích xuất khung hình/âm thanh và gọi Gemma 4 E2B qua Ollama."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "gemma4-e2b-opt",
        whisper_model: str = "small",
        whisper_device: str = "cpu",
        whisper_compute_type: str = "int8",
        max_frames: int = 10,
        image_size: int = 512,
        timeout: int = 180,
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.whisper_model = whisper_model
        self.whisper_device = whisper_device
        self.whisper_compute_type = whisper_compute_type
        self.max_frames = max(1, int(max_frames))
        self.image_size = max(128, int(image_size))
        self.timeout = timeout
        self._whisper: Any = None

    def _encode_frame_bgr(self, frame: np.ndarray, max_side: int) -> str:
        """Resize khung hình giữ nguyên tỷ lệ (cạnh dài <= max_side) và mã hóa base64 JPEG."""
        height, width = frame.shape[:2]
        longest = max(height, width)
        if longest > max_side:
            scale = max_side / float(longest)
            new_w = max(1, int(round(width * scale)))
            new_h = max(1, int(round(height * scale)))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise ValueError("Không thể mã hóa khung hình sang định dạng JPEG.")
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def load_image_base64(self, image_path: str | Path) -> str:
        """Đọc ảnh an toàn với đường dẫn Unicode trên Windows và trả về chuỗi base64."""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {path}")

        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        frame = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Không thể giải mã nội dung file ảnh: {path}")
        return self._encode_frame_bgr(frame, self.image_size)

    def extract_video_frames(self, video_path: str | Path) -> list[str]:
        """Trích xuất tối đa `max_frames` khung hình trải đều từ đầu đến cuối video."""
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file video: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Không thể mở file video: {path}")

        frames_b64: list[str] = []
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                num_samples = min(self.max_frames, total_frames)
                indices = np.linspace(0, total_frames - 1, num=num_samples, dtype=int)
                seen: set[int] = set()
                for idx in indices.tolist():
                    if idx in seen:
                        continue
                    seen.add(idx)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frames_b64.append(self._encode_frame_bgr(frame, self.image_size))
            else:
                # Dự phòng khi container video không báo tổng số frame
                raw_frames: list[np.ndarray] = []
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    raw_frames.append(frame)

                if raw_frames:
                    num_samples = min(self.max_frames, len(raw_frames))
                    indices = np.linspace(0, len(raw_frames) - 1, num=num_samples, dtype=int)
                    for idx in indices.tolist():
                        frames_b64.append(
                            self._encode_frame_bgr(raw_frames[idx], self.image_size)
                        )
        finally:
            cap.release()

        if not frames_b64:
            raise ValueError(f"Video không có khung hình hợp lệ: {path}")
        return frames_b64

    def has_audio_stream(self, video_path: str | Path) -> bool:
        """Kiểm tra file video có chứa luồng âm thanh hay không."""
        try:
            import av
        except ImportError:
            print("[Cảnh báo] Chưa cài thư viện `av`. Bỏ qua bước kiểm tra âm thanh.")
            return False

        try:
            with av.open(str(video_path)) as container:
                return len(container.streams.audio) > 0
        except Exception as exc:
            print(f"[Cảnh báo] Không thể kiểm tra luồng âm thanh của video: {exc}")
            return False

    def _decode_audio_16k_mono(self, video_path: str | Path) -> tuple[bool, np.ndarray]:
        """Tự giải mã âm thanh sang mảng float32 mono 16kHz bằng PyAV (tương thích PyAV 19+)."""
        try:
            import av
        except ImportError:
            print("[Cảnh báo] Chưa cài thư viện `av`. Bỏ qua bước tách âm thanh.")
            return False, np.array([], dtype=np.float32)

        chunks: list[np.ndarray] = []
        try:
            with av.open(str(video_path)) as container:
                if len(container.streams.audio) == 0:
                    return False, np.array([], dtype=np.float32)

                audio_stream = container.streams.audio[0]
                resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)

                for frame in container.decode(audio_stream):
                    resampled = resampler.resample(frame)
                    out_frames = resampled if isinstance(resampled, list) else [resampled]
                    for out_frame in out_frames:
                        if out_frame is not None:
                            chunks.append(out_frame.to_ndarray().reshape(-1))

                # Flush phần buffer còn lại trong resampler
                flushed = resampler.resample(None)
                if flushed is not None:
                    out_frames = flushed if isinstance(flushed, list) else [flushed]
                    for out_frame in out_frames:
                        if out_frame is not None:
                            chunks.append(out_frame.to_ndarray().reshape(-1))
        except Exception as exc:
            print(f"[Cảnh báo] Lỗi khi giải mã âm thanh ({exc}). Tiếp tục phân tích hình ảnh.")
            return True, np.array([], dtype=np.float32)

        if not chunks:
            return True, np.array([], dtype=np.float32)

        audio = np.concatenate(chunks).astype(np.float32, copy=False)
        audio = np.clip(audio, -1.0, 1.0)
        return True, np.ascontiguousarray(audio, dtype=np.float32)

    def transcribe_audio(
        self, video_path: str | Path, language: str | None = "vi"
    ) -> tuple[bool, str]:
        """Chuyển lời thoại trong video sang văn bản bằng faster-whisper (nếu có âm thanh)."""
        has_audio, audio = self._decode_audio_16k_mono(video_path)
        if not has_audio:
            return False, ""
        if audio.size == 0:
            return True, ""

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print(
                "[Cảnh báo] Chưa cài `faster-whisper`. Hệ thống chỉ phân tích phần hình ảnh."
            )
            return True, ""

        try:
            if self._whisper is None:
                self._whisper = WhisperModel(
                    self.whisper_model,
                    device=self.whisper_device,
                    compute_type=self.whisper_compute_type,
                )

            segments, _ = self._whisper.transcribe(
                audio,
                vad_filter=True,
                beam_size=5,
                language=language,
            )
            texts = [seg.text.strip() for seg in segments if seg.text and seg.text.strip()]
            transcript = " ".join(texts).strip()
            return True, transcript
        except Exception as exc:
            print(
                f"[Cảnh báo] Lỗi khi nhận dạng giọng nói ({exc}). Tiếp tục phân tích hình ảnh."
            )
            return True, ""

    def _parse_json_response(self, raw_text: str) -> dict[str, Any]:
        """Giải mã JSON phản hồi từ model, có cơ chế dự phòng nếu output kèm markdown."""
        cleaned = raw_text.strip()
        if not cleaned:
            return {
                "chu_de_chinh": "Chưa xác định",
                "quan_diem_noi_dung": "Chưa xác định",
                "tom_tat_ngan": "",
                "tu_khoa": [],
            }

        candidates = [cleaned]
        fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        if fenced and fenced != cleaned:
            candidates.append(fenced)

        match = re.search(r"\{.*\}", fenced, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    keywords = parsed.get("tu_khoa") or []
                    if isinstance(keywords, str):
                        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
                    return {
                        "chu_de_chinh": str(parsed.get("chu_de_chinh", "Chưa xác định")).strip(),
                        "quan_diem_noi_dung": str(
                            parsed.get("quan_diem_noi_dung", "Chưa xác định")
                        ).strip(),
                        "tom_tat_ngan": str(parsed.get("tom_tat_ngan", "")).strip(),
                        "tu_khoa": [str(k).strip() for k in keywords if str(k).strip()],
                    }
            except json.JSONDecodeError:
                continue

        return {
            "chu_de_chinh": "Chưa xác định",
            "quan_diem_noi_dung": "Chưa xác định",
            "tom_tat_ngan": cleaned,
            "tu_khoa": [],
        }

    def _build_prompt(self, media_type: str, caption: str, transcript: str) -> str:
        """Tạo prompt tiếng Việt hướng dẫn Gemma 4 tổng hợp chủ đề và quan điểm."""
        media_desc = "chuỗi khung hình từ video TikTok" if media_type == "video" else "hình ảnh bài đăng TikTok"
        parts = [
            f"Bạn đang phân tích {media_desc}. Hãy quan sát kỹ chữ hiển thị trên màn hình, hành động, bối cảnh và biểu cảm.",
        ]
        if transcript:
            parts.append(f"Lời thoại trích xuất từ âm thanh video:\n\"\"\"\n{transcript}\n\"\"\"")
        else:
            parts.append("Âm thanh/lời thoại: Không có lời thoại hoặc chỉ có hình ảnh.")

        if caption.strip():
            parts.append(f"Caption/mô tả bài đăng:\n\"\"\"\n{caption.strip()}\n\"\"\"")

        parts.append(
            "Hãy trả về đúng định dạng JSON theo schema với 4 trường tiếng Việt:\n"
            "- chu_de_chinh: Chủ đề chính của nội dung (ngắn gọn, rõ ràng).\n"
            "- quan_diem_noi_dung: Quan điểm, thông điệp hoặc thái độ mà tác giả thể hiện.\n"
            "- tom_tat_ngan: Tóm tắt súc tích nội dung chính trong 2-4 câu.\n"
            "- tu_khoa: Danh sách 3-6 từ khóa liên quan."
        )
        return "\n\n".join(parts)

    def unload_ollama_model(self) -> None:
        """Giải phóng model Gemma 4 khỏi VRAM (hữu ích trước khi chạy DL trên GPU 4GB)."""
        try:
            requests.post(
                f"{self.ollama_url}/api/chat",
                json={"model": self.model_name, "keep_alive": 0},
                timeout=15,
            )
        except requests.RequestException:
            pass

    def _resolve_media_input(
        self, media_input: str | Path | Sequence[str | Path]
    ) -> tuple[str, list[Path]]:
        """Chuẩn hóa đầu vào thành loại media ('video' hoặc 'image') và danh sách đường dẫn."""
        if isinstance(media_input, (str, Path)):
            path = Path(media_input)
            if not path.exists():
                raise FileNotFoundError(f"Không tìm thấy đường dẫn media: {path}")

            if path.is_dir():
                image_files = sorted(
                    p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                )
                if image_files:
                    return "image", image_files[: self.max_frames]

                video_files = sorted(
                    p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
                )
                if video_files:
                    return "video", [video_files[0]]

                raise ValueError(f"Thư mục không chứa file ảnh hoặc video hợp lệ: {path}")

            ext = path.suffix.lower()
            if ext in VIDEO_EXTS:
                return "video", [path]
            if ext in IMAGE_EXTS:
                return "image", [path]
            raise ValueError(f"Định dạng file chưa được hỗ trợ: {path.suffix}")

        paths = [Path(p) for p in media_input]
        if not paths:
            raise ValueError("Danh sách file media đầu vào đang rỗng.")

        if len(paths) == 1 and paths[0].suffix.lower() in VIDEO_EXTS:
            return "video", paths

        for p in paths:
            if p.suffix.lower() not in IMAGE_EXTS:
                raise ValueError(
                    f"Khi truyền nhiều file cùng lúc, tất cả phải là file ảnh hợp lệ. File lỗi: {p}"
                )
        return "image", paths[: self.max_frames]

    def analyze(
        self,
        media_input: str | Path | Sequence[str | Path],
        caption: str = "",
        enable_audio: bool = True,
        language: str | None = "vi",
        think: bool = False,
    ) -> MediaAnalysisResult:
        """Phân tích video hoặc hình ảnh TikTok bằng Gemma 4 E2B."""
        media_type, paths = self._resolve_media_input(media_input)

        has_audio = False
        transcript = ""
        if media_type == "video":
            video_path = paths[0]
            images_b64 = self.extract_video_frames(video_path)
            if enable_audio:
                has_audio, transcript = self.transcribe_audio(video_path, language=language)
        else:
            images_b64 = [self.load_image_base64(p) for p in paths]

        prompt = self._build_prompt(
            media_type=media_type,
            caption=caption,
            transcript=transcript,
        )

        payload = {
            "model": self.model_name,
            "stream": False,
            "think": think,
            "format": MEDIA_RESPONSE_SCHEMA,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": images_b64,
                }
            ],
        }

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Không kết nối được tới Ollama tại {self.ollama_url}. "
                "Hãy kiểm tra container Docker `gemma4-e2b` đã khởi động chưa "
                "(chạy `docker compose up -d` trong thư mục AI_Local/gemma4-e2b)."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            raise RuntimeError(
                f"Ollama trả về lỗi khi phân tích với model `{self.model_name}`: {detail}"
            ) from exc

        raw_content = resp.json().get("message", {}).get("content", "")
        parsed = self._parse_json_response(raw_content)

        return MediaAnalysisResult(
            media_type=media_type,
            source_paths=[str(p) for p in paths],
            num_frames_used=len(images_b64),
            has_audio=has_audio,
            transcript=transcript,
            chu_de_chinh=parsed["chu_de_chinh"],
            quan_diem_noi_dung=parsed["quan_diem_noi_dung"],
            tom_tat_ngan=parsed["tom_tat_ngan"],
            tu_khoa=parsed["tu_khoa"],
            raw_response=raw_content,
        )
