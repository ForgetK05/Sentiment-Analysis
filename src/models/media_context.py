from __future__ import annotations

import base64
import gc
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import requests

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _natural_sort_key(path: Path) -> list[int | str]:
    """Khóa sắp xếp tự nhiên để 1.jpg, 2.jpg, ..., 10.jpg đúng thứ tự."""
    return [
        int(tok) if tok.isdigit() else tok.lower()
        for tok in re.split(r"(\d+)", path.name)
    ]


def _clean_whisper_prompt(caption: str) -> str | None:
    """Loại bỏ URL, #hashtag, @mention và emoji khỏi caption trước khi mồi Whisper."""
    if not caption or not caption.strip():
        return None
    text = re.sub(r"https?://\S+|www\.\S+", " ", caption)
    text = re.sub(r"[#@]\S+", " ", text)
    text = re.sub(r"[^\w\s,.;:!?\'\"()\-/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:200] if text else None


MEDIA_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chu_tren_man_hinh": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Trích nguyên văn theo trình tự các chữ, tiêu đề, phụ đề, giá tiền, "
                "tên sản phẩm/quán/nhân vật, số liệu xuất hiện trên các khung hình hoặc ảnh."
            ),
        },
        "dien_bien_hinh_anh": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Mô tả cụ thể những gì xảy ra theo trình tự thời gian (từ F1 đến ảnh cuối): "
                "ai xuất hiện, làm hành động gì, bối cảnh/đồ vật gì, biểu cảm, cảnh so sánh hoặc cú twist."
            ),
        },
        "chi_tiet_va_luan_diem": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Các sự kiện, lập luận, lời khen/chê, con số, vấn đề cụ thể hoặc câu thoại đắt giá "
                "tổng hợp từ cả chữ trên ảnh, hình ảnh và lời thoại."
            ),
        },
        "tom_tat_ngan": {
            "type": "string",
            "description": (
                "Bản tóm tắt NỘI DUNG CỐT LÕI (3-5 câu) nêu thẳng video đang nói về chuyện gì cụ thể: "
                "[Ai/Cái gì] + [Sự việc/Vấn đề/Luận điểm cụ thể] + [Chi tiết/Số liệu/Bằng chứng chính] + "
                "[Kết quả/Cú twist/Thông điệp cuối]. TUYỆT ĐỐI KHÔNG viết chung chung."
            ),
        },
        "chu_de_chinh": {
            "type": "string",
            "description": "Chủ đề cụ thể của video/bài đăng (nêu rõ lĩnh vực và đối tượng).",
        },
        "muc_dich_chinh": {
            "type": "string",
            "description": (
                "Mục đích chính: Giải trí/Hài hước, Bóc phốt/Chỉ trích/Drama, "
                "Đánh giá/Review sản phẩm, Chia sẻ thông tin/Tin tức, Tâm sự/Cảm xúc cá nhân, "
                "Bán hàng/Quảng cáo, hoặc Tranh luận quan điểm."
            ),
        },
        "quan_diem_noi_dung": {
            "type": "string",
            "description": "Quan điểm, thái độ hoặc kết luận cụ thể mà tác giả muốn truyền tải.",
        },
        "doi_tuong_chinh": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tên cụ thể nhân vật, thương hiệu, sản phẩm, địa điểm hoặc sự kiện trung tâm.",
        },
        "sac_thai_noi_dung": {
            "type": "string",
            "enum": [
                "Tích cực",
                "Tiêu cực",
                "Châm biếm / Mỉa mai",
                "Tranh cãi",
                "Trung lập",
            ],
            "description": "Sắc thái chủ đạo của bài đăng.",
        },
        "tu_khoa": {
            "type": "array",
            "items": {"type": "string"},
            "description": "4-8 từ khóa trọng tâm (gồm cả tên riêng, tiếng lóng, viết tắt nếu có).",
        },
        "luu_y_danh_gia_comment": {
            "type": "string",
            "description": (
                "Chỉ rõ chi tiết nào trong video sẽ khiến người xem khen, chê, tranh cãi hay cà khịa "
                "để phân biệt comment đang đồng tình với tác giả hay công kích tác giả."
            ),
        },
    },
    "required": [
        "chu_tren_man_hinh",
        "dien_bien_hinh_anh",
        "chi_tiet_va_luan_diem",
        "tom_tat_ngan",
        "chu_de_chinh",
        "muc_dich_chinh",
        "quan_diem_noi_dung",
        "doi_tuong_chinh",
        "sac_thai_noi_dung",
        "tu_khoa",
        "luu_y_danh_gia_comment",
    ],
}

SLIDE_BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ocr_texts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Toàn bộ chữ, tiêu đề, giá tiền, số liệu, tên riêng đọc được trên các ảnh trong lô này.",
        },
        "visual_notes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Mô tả ngắn gọn từng ảnh theo đúng nhãn Anh i/N: nhân vật, hành động, chi tiết chính.",
        },
    },
    "required": ["ocr_texts", "visual_notes"],
}

_SYSTEM_PROMPT = (
    "Bạn là chuyên gia phân tích chi tiết nội dung video/hình ảnh TikTok phục vụ hệ thống Social Listening song ngữ Việt – Anh.\n"
    "QUY TẮC BẮT BUỘC:\n"
    "1. TRÍCH XUẤT BẰNG CHỨNG TRƯỚC KHI KẾT LUẬN:\n"
    "   - Đọc thật kỹ mọi dòng chữ trên từng ảnh/frame (`chu_tren_man_hinh`), không bỏ sót tiêu đề, phụ đề, mức giá, tên sản phẩm/nhân vật, đoạn chat hay chữ nhỏ ở góc.\n"
    "   - Ghi lại diễn biến theo đúng thứ tự các khung hình F1 -> F cuối hoặc Anh 1 -> Anh cuối (`dien_bien_hinh_anh`), đối chiếu mốc thời gian trên ảnh với mốc thời gian trong lời thoại Whisper.\n"
    "   - Whisper có thể nghe nhầm tên riêng hoặc bị nhiễu do nhạc nền: luôn lấy CHỮ TRÊN MÀN HÌNH và CAPTION làm nguồn đối chiếu chính.\n"
    "   - Không tự bịa thông tin không xuất hiện trong ảnh, lời thoại hoặc caption. Nếu chữ mờ không đọc được, bỏ qua hoặc ghi \"không nhìn rõ\".\n"
    "2. CÁCH VIẾT `tom_tat_ngan` (QUAN TRỌNG NHẤT — TÓM TẮT ĐÚNG NỘI DUNG CHÍNH):\n"
    "   - CẤM TUYỆT ĐỐI các câu mở đầu chung chung, sáo rỗng như: \"Video nói về...\", \"Nội dung xoay quanh...\", \"Người đăng chia sẻ trải nghiệm về...\", \"Video bàn luận về một vấn đề...\".\n"
    "   - BẮT BUỘC đi thẳng vào sự việc cụ thể theo 4 phần:\n"
    "     (1) Ai / Sản phẩm / Sự kiện cụ thể nào ->\n"
    "     (2) Chuyện gì đang xảy ra hoặc tác giả nêu luận điểm/vấn đề cụ thể gì ->\n"
    "     (3) Các chi tiết, con số, hành động, bằng chứng nổi bật trong video ->\n"
    "     (4) Kết quả, cú twist cuối video hoặc thông điệp/kết luận của tác giả.\n"
    "   - Người đọc chỉ cần đọc `tom_tat_ngan` là nắm trọn nội dung chính mà không cần mở xem lại video.\n"
    "3. PHỤC VỤ HIỂU BÌNH LUẬN:\n"
    "   - Trong `luu_y_danh_gia_comment`, chỉ rõ: nếu người xem chê/chửi thì họ đang chê nhân vật/sản phẩm trong video (tức đồng tình với tác giả) hay đang chê chính tác giả; nêu rõ các chi tiết dễ gây tranh cãi hoặc meme/câu đùa ở cuối video.\n"
    "   - Viết tiếng Việt tự nhiên, rõ ràng, giữ nguyên tên riêng và thuật ngữ/từ lóng tiếng Anh gốc.\n"
    "CHỈ trả về duy nhất một đối tượng JSON hợp lệ khớp đúng schema, không thêm lời dẫn hay markdown."
)


@dataclass
class MediaAnalysisResult:
    """Kết quả phân tích nội dung đa phương tiện (video/ảnh) từ Gemma 4 E2B."""

    media_type: str
    source_paths: list[str]
    num_frames_used: int
    has_audio: bool
    caption: str
    transcript: str
    chu_de_chinh: str
    muc_dich_chinh: str
    quan_diem_noi_dung: str
    doi_tuong_chinh: list[str] = field(default_factory=list)
    sac_thai_noi_dung: str = "Trung lập"
    tom_tat_ngan: str = ""
    tu_khoa: list[str] = field(default_factory=list)
    luu_y_danh_gia_comment: str = ""
    raw_response: str = ""
    chu_tren_man_hinh: list[str] = field(default_factory=list)
    dien_bien_hinh_anh: list[str] = field(default_factory=list)
    chi_tiet_va_luan_diem: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MediaAnalyzer:
    """Giai đoạn 1 (AI Đầu vào): Trích xuất khung hình/âm thanh và gọi Gemma 4 E2B qua Ollama."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "gemma4-e2b-opt",
        whisper_model: str = "small",
        whisper_device: str = "cpu",
        whisper_compute_type: str = "int8",
        max_frames: int = 8,
        image_size: int = 768,
        temperature: float = 0.1,
        num_ctx: int = 16384,
        repeat_penalty: float = 1.1,
        enable_two_pass_slideshow: bool = True,
        two_pass_batch_size: int = 4,
        timeout: int = 180,
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.whisper_model = whisper_model
        self.whisper_device = whisper_device
        self.whisper_compute_type = whisper_compute_type
        self.max_frames = max(1, int(max_frames))
        self.image_size = max(128, int(image_size))
        self.temperature = float(temperature)
        self.num_ctx = max(2048, int(num_ctx))
        self.repeat_penalty = float(repeat_penalty)
        self.enable_two_pass_slideshow = bool(enable_two_pass_slideshow)
        self.two_pass_batch_size = max(1, int(two_pass_batch_size))
        self.timeout = timeout
        self._whisper: Any = None

    @staticmethod
    def _resize_frame_bgr(frame: np.ndarray, max_side: int) -> np.ndarray:
        height, width = frame.shape[:2]
        longest = max(height, width)
        if longest > max_side:
            scale = max_side / float(longest)
            new_w = max(1, int(round(width * scale)))
            new_h = max(1, int(round(height * scale)))
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return frame

    @staticmethod
    def _stamp_frame_label(frame: np.ndarray, label: str) -> np.ndarray:
        """Đóng dấu nhãn mốc thời gian / thứ tự ảnh sau khi resize để chữ luôn sắc nét."""
        annotated = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.55, min(annotated.shape[0], annotated.shape[1]) / 850.0)
        thickness = max(1, int(round(scale * 2)))
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        cv2.rectangle(annotated, (8, 8), (18 + tw, 18 + th + baseline), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            label,
            (13, 13 + th),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        return annotated

    def _encode_frame_bgr(
        self,
        frame: np.ndarray,
        max_side: int,
        label: str | None = None,
    ) -> str:
        resized = self._resize_frame_bgr(frame, max_side)
        if label:
            resized = self._stamp_frame_label(resized, label)
        ok, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise ValueError("Không thể mã hóa khung hình sang định dạng JPEG.")
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def load_image_base64(
        self,
        image_path: str | Path,
        label: str | None = None,
    ) -> str:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {path}")

        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        frame = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Không thể giải mã nội dung file ảnh: {path}")
        return self._encode_frame_bgr(frame, self.image_size, label=label)

    def _read_video_sequential_safe(
        self,
        path: Path,
        fps: float,
    ) -> list[tuple[int, np.ndarray]]:
        """Đọc tuần tự và resize ngay từng frame để chống tràn RAM với video VFR hoặc thiếu metadata."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return []
        sampled: list[tuple[int, np.ndarray]] = []
        stride = 15
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if frame_idx % stride == 0:
                    pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                    sec = (
                        int(round(pos_msec / 1000.0))
                        if (pos_msec > 0.0 or frame_idx == 0)
                        else int(round(frame_idx / fps))
                    )
                    sampled.append((sec, self._resize_frame_bgr(frame, self.image_size)))
                    if len(sampled) > 64:
                        sampled = sampled[::2]
                        stride *= 2
                frame_idx += 1
        finally:
            cap.release()
        return sampled

    def extract_video_frames(self, video_path: str | Path) -> list[str]:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file video: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Không thể mở file video: {path}")

        sampled_pairs: list[tuple[int, np.ndarray]] = []
        fps = 30.0
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            fps = raw_fps if raw_fps > 0.0 else 30.0

            if total_frames > 0:
                num_samples = min(self.max_frames, total_frames)
                start_idx = int(total_frames * 0.03) if total_frames > 30 else 0
                end_idx = int(total_frames * 0.95) if total_frames > 30 else total_frames - 1
                indices = sorted(
                    set(
                        np.linspace(
                            start_idx,
                            max(start_idx, end_idx),
                            num=num_samples,
                            dtype=int,
                        ).tolist()
                    )
                )
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                        sec = (
                            int(round(pos_msec / 1000.0))
                            if (pos_msec > 0.0 or idx == 0)
                            else int(round(idx / fps))
                        )
                        sampled_pairs.append((sec, self._resize_frame_bgr(frame, self.image_size)))
        finally:
            cap.release()

        # Fallback an toàn nếu thiếu total_frames hoặc video VFR seek thất bại
        if len(sampled_pairs) < min(2, self.max_frames):
            seq_pairs = self._read_video_sequential_safe(path, fps=fps)
            if seq_pairs:
                if len(seq_pairs) <= self.max_frames:
                    sampled_pairs = seq_pairs
                else:
                    pick = np.linspace(
                        0,
                        len(seq_pairs) - 1,
                        num=self.max_frames,
                        dtype=int,
                    ).tolist()
                    sampled_pairs = [seq_pairs[pos] for pos in pick]

        if not sampled_pairs:
            raise ValueError(f"Video không có khung hình hợp lệ: {path}")

        frames_b64: list[str] = []
        for i, (sec, resized) in enumerate(sampled_pairs, start=1):
            label = f"F{i} {sec // 60:02d}:{sec % 60:02d}"
            frames_b64.append(self._encode_frame_bgr(resized, self.image_size, label=label))
        return frames_b64

    def has_audio_stream(self, video_path: str | Path) -> bool:
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
        self,
        video_path: str | Path,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> tuple[bool, str]:
        """Chuyển lời thoại trong video sang văn bản kèm mốc thời gian [MM:SS-MM:SS] bằng faster-whisper."""
        has_audio, audio = self._decode_audio_16k_mono(video_path)
        if not has_audio:
            return False, ""
        if audio.size == 0:
            return True, ""

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("[Cảnh báo] Chưa cài `faster-whisper`. Hệ thống chỉ phân tích phần hình ảnh.")
            return True, ""

        try:
            if self._whisper is None:
                self._whisper = WhisperModel(
                    self.whisper_model,
                    device=self.whisper_device,
                    compute_type=self.whisper_compute_type,
                )

            clean_prompt = _clean_whisper_prompt(initial_prompt or "")
            transcribe_kwargs: dict[str, Any] = {
                "vad_filter": True,
                "beam_size": 5,
                "condition_on_previous_text": False,
                "no_speech_threshold": 0.6,
                "compression_ratio_threshold": 2.4,
            }
            if language:
                transcribe_kwargs["language"] = language
            if clean_prompt:
                transcribe_kwargs["initial_prompt"] = clean_prompt

            segments, _ = self._whisper.transcribe(audio, **transcribe_kwargs)
            lines: list[str] = []
            for seg in segments:
                text = (seg.text or "").strip()
                if text and getattr(seg, "no_speech_prob", 0.0) < 0.7:
                    s_m, s_s = divmod(int(seg.start), 60)
                    e_m, e_s = divmod(int(seg.end), 60)
                    lines.append(f"[{s_m:02d}:{s_s:02d}-{e_m:02d}:{e_s:02d}] {text}")
            transcript = "\n".join(lines).strip()
            return True, transcript
        except Exception as exc:
            print(f"[Cảnh báo] Lỗi khi nhận dạng giọng nói ({exc}). Tiếp tục phân tích hình ảnh.")
            return True, ""

    @staticmethod
    def _coerce_str_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [k.strip() for k in value.split(",") if k.strip()]
        return []

    def _parse_json_response(self, raw_text: str) -> dict[str, Any]:
        default_payload: dict[str, Any] = {
            "chu_tren_man_hinh": [],
            "dien_bien_hinh_anh": [],
            "chi_tiet_va_luan_diem": [],
            "chu_de_chinh": "Chưa xác định",
            "muc_dich_chinh": "Chưa xác định",
            "quan_diem_noi_dung": "Chưa xác định",
            "doi_tuong_chinh": [],
            "sac_thai_noi_dung": "Trung lập",
            "tom_tat_ngan": "",
            "tu_khoa": [],
            "luu_y_danh_gia_comment": "",
        }
        cleaned = (raw_text or "").strip()
        if not cleaned:
            return default_payload

        candidates = [cleaned]
        fenced = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL
        ).strip()
        if fenced and fenced != cleaned:
            candidates.append(fenced)

        match = re.search(r"\{.*\}", fenced, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    tone = str(parsed.get("sac_thai_noi_dung", "Trung lập")).strip()
                    valid_tones = {
                        "Tích cực",
                        "Tiêu cực",
                        "Châm biếm / Mỉa mai",
                        "Tranh cãi",
                        "Trung lập",
                    }
                    if tone not in valid_tones:
                        tone = "Trung lập"
                    return {
                        "chu_tren_man_hinh": self._coerce_str_list(parsed.get("chu_tren_man_hinh")),
                        "dien_bien_hinh_anh": self._coerce_str_list(parsed.get("dien_bien_hinh_anh")),
                        "chi_tiet_va_luan_diem": self._coerce_str_list(parsed.get("chi_tiet_va_luan_diem")),
                        "chu_de_chinh": str(parsed.get("chu_de_chinh", "Chưa xác định")).strip(),
                        "muc_dich_chinh": str(parsed.get("muc_dich_chinh", "Chưa xác định")).strip(),
                        "quan_diem_noi_dung": str(
                            parsed.get("quan_diem_noi_dung", "Chưa xác định")
                        ).strip(),
                        "doi_tuong_chinh": self._coerce_str_list(parsed.get("doi_tuong_chinh")),
                        "sac_thai_noi_dung": tone,
                        "tom_tat_ngan": str(parsed.get("tom_tat_ngan", "")).strip(),
                        "tu_khoa": self._coerce_str_list(parsed.get("tu_khoa")),
                        "luu_y_danh_gia_comment": str(
                            parsed.get("luu_y_danh_gia_comment", "")
                        ).strip(),
                    }
            except json.JSONDecodeError:
                continue

        # Không gán chuỗi JSON lỗi vào tom_tat_ngan để tránh làm nhiễu gemma_auditor
        return default_payload

    def _build_messages(
        self,
        media_type: str,
        caption: str,
        transcript: str,
        images_b64: list[str],
        slide_notes: str = "",
    ) -> list[dict[str, Any]]:
        media_desc = (
            f"chuỗi {len(images_b64)} khung hình có đóng dấu mốc thời gian (F1..F{len(images_b64)}) từ video TikTok"
            if media_type == "video"
            else f"{len(images_b64)} hình ảnh có đánh số thứ tự của bài đăng TikTok"
        )

        sections = [
            f"Dữ liệu đầu vào gồm {media_desc}. Hãy đọc kỹ từng chữ trên màn hình, "
            "bối cảnh, biểu cảm nhân vật và trình tự diễn biến."
        ]

        if caption.strip():
            sections.append(f"Caption / Hashtags bài đăng:\n\"\"\"\n{caption.strip()}\n\"\"\"")
        else:
            sections.append("Caption / Hashtags bài đăng: (Không có).")

        if transcript.strip():
            sections.append(
                "Lời thoại có mốc thời gian trích xuất từ âm thanh (Whisper ASR — có thể sai tên riêng, "
                "hãy đối chiếu với chữ trên khung hình và caption để hiệu chỉnh ngữ nghĩa):\n"
                f"\"\"\"\n{transcript.strip()}\n\"\"\""
            )
        else:
            sections.append("Âm thanh / lời thoại: Không có lời thoại hoặc video chỉ có nhạc nền/hình ảnh.")

        if slide_notes.strip():
            sections.append(
                "Ghi chú chi tiết đã đọc từ toàn bộ các slide của bài đăng (Pass 1):\n"
                f"\"\"\"\n{slide_notes.strip()}\n\"\"\""
            )

        sections.append(
            "Hãy phân tích theo đúng trình tự Evidence-First và trả về ĐÚNG JSON khớp 11 trường của schema:\n"
            "1. chu_tren_man_hinh -> 2. dien_bien_hinh_anh -> 3. chi_tiet_va_luan_diem -> "
            "4. tom_tat_ngan (3-5 câu nêu thẳng sự việc cụ thể, cấm mở đầu chung chung) -> "
            "5. chu_de_chinh -> 6. muc_dich_chinh -> 7. quan_diem_noi_dung -> 8. doi_tuong_chinh -> "
            "9. sac_thai_noi_dung -> 10. tu_khoa -> 11. luu_y_danh_gia_comment."
        )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "\n\n".join(sections),
                "images": images_b64,
            },
        ]

    def _call_ollama_chat(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        think: bool = False,
    ) -> str:
        payload = {
            "model": self.model_name,
            "stream": False,
            "think": think,
            "format": schema,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
                "num_ctx": self.num_ctx,
                "repeat_penalty": self.repeat_penalty,
            },
            "messages": messages,
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

        return str(resp.json().get("message", {}).get("content", "") or "")

    def unload_whisper_model(self) -> None:
        """Giải phóng Faster-Whisper khỏi RAM trước khi nạp PhoBERT/ViSoBERT."""
        if self._whisper is not None:
            self._whisper = None
            gc.collect()

    def unload_ollama_model(self) -> None:
        """Giải phóng model Gemma 4 khỏi VRAM trước khi chạy các mô hình NLP/DL."""
        try:
            requests.post(
                f"{self.ollama_url}/api/chat",
                json={"model": self.model_name, "keep_alive": 0},
                timeout=15,
            )
        except requests.RequestException:
            pass

    def _sample_image_paths(self, paths: list[Path], limit: int | None = None) -> list[Path]:
        cap = self.max_frames if limit is None else max(1, limit)
        if len(paths) <= cap:
            return paths
        indices = np.linspace(0, len(paths) - 1, num=cap, dtype=int)
        return [paths[idx] for idx in indices.tolist()]

    def _resolve_media_input(
        self,
        media_input: str | Path | Sequence[str | Path],
        sample: bool = True,
    ) -> tuple[str, list[Path]]:
        if isinstance(media_input, (str, Path)):
            path = Path(media_input)
            if not path.exists():
                raise FileNotFoundError(f"Không tìm thấy đường dẫn media: {path}")

            if path.is_dir():
                image_files = sorted(
                    (p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
                    key=_natural_sort_key,
                )
                if image_files:
                    chosen = self._sample_image_paths(image_files) if sample else image_files
                    return "image", chosen

                video_files = sorted(
                    (p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS),
                    key=_natural_sort_key,
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
        chosen = self._sample_image_paths(paths) if sample else paths
        return "image", chosen

    def _extract_slideshow_two_pass_notes(
        self,
        all_paths: list[Path],
        think: bool = False,
    ) -> str:
        """Pass 1 cho slideshow dài (> max_frames): đọc OCR và mô tả ngắn theo từng lô ảnh."""
        total = len(all_paths)
        note_blocks: list[str] = []
        for start in range(0, total, self.two_pass_batch_size):
            batch = all_paths[start : start + self.two_pass_batch_size]
            batch_b64 = [
                self.load_image_base64(p, label=f"Anh {start + i + 1}/{total}")
                for i, p in enumerate(batch)
            ]
            end_idx = start + len(batch)
            prompt = (
                f"Đây là nhóm ảnh {start + 1}..{end_idx} trong tổng số {total} ảnh của bài slideshow TikTok.\n"
                "Hãy trích xuất nguyên văn toàn bộ chữ viết (`ocr_texts`) và mô tả ngắn từng ảnh theo nhãn (`visual_notes`)."
            )
            raw = self._call_ollama_chat(
                messages=[
                    {"role": "system", "content": "Trích xuất chính xác chữ trên ảnh và diễn biến ngắn theo đúng JSON schema."},
                    {"role": "user", "content": prompt, "images": batch_b64},
                ],
                schema=SLIDE_BATCH_SCHEMA,
                think=think,
            )
            try:
                parsed = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                parsed = {}
            ocr = self._coerce_str_list(parsed.get("ocr_texts"))
            vis = self._coerce_str_list(parsed.get("visual_notes"))
            lines = [f"[Lô Anh {start + 1}-{end_idx}]"]
            if ocr:
                lines.append("Chữ trên ảnh: " + " | ".join(ocr))
            if vis:
                lines.append("Hình ảnh: " + " ; ".join(vis))
            if len(lines) > 1:
                note_blocks.append("\n".join(lines))
        return "\n\n".join(note_blocks)

    def analyze(
        self,
        media_input: str | Path | Sequence[str | Path],
        caption: str = "",
        enable_audio: bool = True,
        language: str | None = None,
        think: bool = False,
    ) -> MediaAnalysisResult:
        """Phân tích video hoặc hình ảnh TikTok bằng Gemma 4 E2B."""
        media_type, all_paths = self._resolve_media_input(media_input, sample=False)

        has_audio = False
        transcript = ""
        slide_notes = ""
        used_paths = all_paths

        if media_type == "video":
            video_path = all_paths[0]
            images_b64 = self.extract_video_frames(video_path)
            if enable_audio:
                whisper_prompt = caption.strip() if caption.strip() else None
                has_audio, transcript = self.transcribe_audio(
                    video_path,
                    language=language,
                    initial_prompt=whisper_prompt,
                )
            num_frames_used = len(images_b64)
        else:
            total = len(all_paths)
            if self.enable_two_pass_slideshow and total > self.max_frames:
                slide_notes = self._extract_slideshow_two_pass_notes(all_paths, think=think)
                sampled_paths = self._sample_image_paths(all_paths, limit=min(4, self.max_frames))
                index_map = {p: idx + 1 for idx, p in enumerate(all_paths)}
                images_b64 = [
                    self.load_image_base64(p, label=f"Anh {index_map[p]}/{total}")
                    for p in sampled_paths
                ]
                num_frames_used = total
            else:
                sampled_paths = self._sample_image_paths(all_paths)
                used_paths = sampled_paths
                images_b64 = [
                    self.load_image_base64(p, label=f"Anh {i}/{len(sampled_paths)}")
                    for i, p in enumerate(sampled_paths, start=1)
                ]
                num_frames_used = len(images_b64)

        messages = self._build_messages(
            media_type=media_type,
            caption=caption,
            transcript=transcript,
            images_b64=images_b64,
            slide_notes=slide_notes,
        )

        raw_content = self._call_ollama_chat(
            messages=messages,
            schema=MEDIA_RESPONSE_SCHEMA,
            think=think,
        )
        parsed = self._parse_json_response(raw_content)

        return MediaAnalysisResult(
            media_type=media_type,
            source_paths=[str(p) for p in used_paths],
            num_frames_used=num_frames_used,
            has_audio=has_audio,
            caption=caption.strip(),
            transcript=transcript,
            chu_de_chinh=parsed["chu_de_chinh"],
            muc_dich_chinh=parsed["muc_dich_chinh"],
            quan_diem_noi_dung=parsed["quan_diem_noi_dung"],
            doi_tuong_chinh=parsed["doi_tuong_chinh"],
            sac_thai_noi_dung=parsed["sac_thai_noi_dung"],
            tom_tat_ngan=parsed["tom_tat_ngan"],
            tu_khoa=parsed["tu_khoa"],
            luu_y_danh_gia_comment=parsed["luu_y_danh_gia_comment"],
            raw_response=raw_content,
            chu_tren_man_hinh=parsed["chu_tren_man_hinh"],
            dien_bien_hinh_anh=parsed["dien_bien_hinh_anh"],
            chi_tiet_va_luan_diem=parsed["chi_tiet_va_luan_diem"],
        )
