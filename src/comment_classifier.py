from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from tqdm import tqdm

DEFAULT_MODEL_NAME = "5CD-AI/Vietnamese-Sentiment-visobert"

LABEL_VI_MAP: dict[str, str] = {
    "POS": "Tích cực",
    "NEG": "Tiêu cực",
    "NEU": "Trung tính",
}

# Chuẩn hóa nhẹ các từ viết tắt/teencode phổ biến trên TikTok Việt Nam
# mà vẫn giữ nguyên emoji để ViSoBERT khai thác tín hiệu cảm xúc.
TEENCODE_MAP: dict[str, str] = {
    "ko": "không",
    "k": "không",
    "kh": "không",
    "khum": "không",
    "hong": "không",
    "hem": "không",
    "dc": "được",
    "đc": "được",
    "mn": "mọi người",
    "sp": "sản phẩm",
    "cx": "cũng",
    "vs": "với",
    "ntn": "như thế nào",
    "đr": "đúng rồi",
    "mik": "mình",
    "mk": "mình",
    "bt": "bình thường",
    "oke": "ok",
    "okela": "ok",
    "j": "gì",
    "z": "vậy",
    "v": "vậy",
    "wa": "quá",
    "wá": "quá",
    "iu": "yêu",
}

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_REPEAT_CHAR_PATTERN = re.compile(r"(.)\1{3,}")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_TEENCODE_PATTERN = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(TEENCODE_MAP, key=len, reverse=True)) + r")(?!\w)",
    flags=re.IGNORECASE,
)


def preprocess_comment(text: str, normalize_teencode: bool = True) -> str:
    """Làm sạch bình luận tiếng Việt MXH nhưng vẫn giữ lại emoji và dấu câu biểu cảm."""
    if text is None:
        return ""
    cleaned = str(text).strip()
    if not cleaned:
        return ""

    cleaned = _URL_PATTERN.sub(" ", cleaned)
    cleaned = _REPEAT_CHAR_PATTERN.sub(r"\1\1", cleaned)
    if normalize_teencode:
        cleaned = _TEENCODE_PATTERN.sub(
            lambda m: TEENCODE_MAP.get(m.group(1).lower(), m.group(1)),
            cleaned,
        )
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


@dataclass
class CommentPrediction:
    """Kết quả phân loại cảm xúc cho một bình luận."""

    original_text: str
    cleaned_text: str
    label: str
    label_vi: str
    confidence: float
    scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU approximation (Abramowitz & Stegun erf, sai số < 1.5e-7, không cần SciPy)."""
    z = x * np.float32(0.7071067811865475)
    sign = np.sign(z)
    az = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * az)
    erf_z = sign * (
        1.0
        - (
            ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
        )
        * np.exp(-az * az)
    )
    return np.float32(0.5) * x * (np.float32(1.0) + erf_z.astype(np.float32))


def _layer_norm(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float
) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(var + np.float32(eps))
    return normed * weight + bias


def _linear(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None
) -> np.ndarray:
    out = np.matmul(x, weight.T)
    if bias is not None:
        out = out + bias
    return out


class _NumpyXLMRobertaClassifier:
    """Suy luận XLMRobertaForSequenceClassification thuần NumPy từ config.json + model.safetensors."""

    def __init__(self, model_name_or_path: str) -> None:
        from safetensors.numpy import load_file

        local_dir = Path(model_name_or_path)
        if local_dir.is_dir():
            config_path = local_dir / "config.json"
            weights_path = local_dir / "model.safetensors"
        else:
            from huggingface_hub import hf_hub_download

            config_path = Path(hf_hub_download(repo_id=model_name_or_path, filename="config.json"))
            weights_path = Path(
                hf_hub_download(repo_id=model_name_or_path, filename="model.safetensors")
            )

        with config_path.open("r", encoding="utf-8") as f:
            self.config: dict[str, Any] = json.load(f)

        raw_weights = load_file(str(weights_path))
        self.weights: dict[str, np.ndarray] = {
            k: v.astype(np.float32, copy=False) for k, v in raw_weights.items()
        }

        self.pad_token_id = int(self.config.get("pad_token_id", 1))
        self.layer_norm_eps = float(self.config.get("layer_norm_eps", 1e-5))
        self.num_hidden_layers = int(self.config["num_hidden_layers"])
        self.num_attention_heads = int(self.config["num_attention_heads"])
        self.hidden_size = int(self.config["hidden_size"])
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.prefix = (
            "roberta."
            if "roberta.embeddings.word_embeddings.weight" in self.weights
            else ""
        )

    def _ln_params(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        w = self.weights.get(f"{name}.weight")
        if w is None:
            w = self.weights[f"{name}.gamma"]
        b = self.weights.get(f"{name}.bias")
        if b is None:
            b = self.weights[f"{name}.beta"]
        return w, b

    def predict_probs(
        self, input_ids: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        """Tính xác suất 3 nhãn cảm xúc cho batch token IDs."""
        p = self.prefix
        mask = (input_ids != self.pad_token_id).astype(np.int64)
        position_ids = (np.cumsum(mask, axis=1) * mask) + self.pad_token_id

        word_emb = self.weights[f"{p}embeddings.word_embeddings.weight"][input_ids]
        pos_emb = self.weights[f"{p}embeddings.position_embeddings.weight"][position_ids]
        tok_type_emb = self.weights[f"{p}embeddings.token_type_embeddings.weight"][0]
        hidden = word_emb + pos_emb + tok_type_emb

        emb_ln_w, emb_ln_b = self._ln_params(f"{p}embeddings.LayerNorm")
        hidden = _layer_norm(hidden, emb_ln_w, emb_ln_b, self.layer_norm_eps)

        batch_size, seq_len = input_ids.shape
        ext_mask = (
            np.float32(1.0) - attention_mask[:, None, None, :].astype(np.float32)
        ) * np.float32(-1e9)
        inv_sqrt_d = np.float32(1.0 / np.sqrt(float(self.head_dim)))

        for idx in range(self.num_hidden_layers):
            lp = f"{p}encoder.layer.{idx}"
            ap = f"{lp}.attention.self"

            q = (
                _linear(hidden, self.weights[f"{ap}.query.weight"], self.weights.get(f"{ap}.query.bias"))
                .reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            k = (
                _linear(hidden, self.weights[f"{ap}.key.weight"], self.weights.get(f"{ap}.key.bias"))
                .reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            v = (
                _linear(hidden, self.weights[f"{ap}.value.weight"], self.weights.get(f"{ap}.value.bias"))
                .reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )

            attn_scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * inv_sqrt_d + ext_mask
            attn_probs = _softmax(attn_scores, axis=-1)
            context = (
                np.matmul(attn_probs, v)
                .transpose(0, 2, 1, 3)
                .reshape(batch_size, seq_len, self.hidden_size)
            )

            ao = f"{lp}.attention.output"
            attn_out = _linear(context, self.weights[f"{ao}.dense.weight"], self.weights.get(f"{ao}.dense.bias"))
            ao_w, ao_b = self._ln_params(f"{ao}.LayerNorm")
            hidden = _layer_norm(hidden + attn_out, ao_w, ao_b, self.layer_norm_eps)

            inter = _gelu(
                _linear(
                    hidden,
                    self.weights[f"{lp}.intermediate.dense.weight"],
                    self.weights.get(f"{lp}.intermediate.dense.bias"),
                )
            )
            ffn_out = _linear(
                inter,
                self.weights[f"{lp}.output.dense.weight"],
                self.weights.get(f"{lp}.output.dense.bias"),
            )
            out_w, out_b = self._ln_params(f"{lp}.output.LayerNorm")
            hidden = _layer_norm(hidden + ffn_out, out_w, out_b, self.layer_norm_eps)

        cls_hidden = hidden[:, 0, :]
        pooled = np.tanh(
            _linear(
                cls_hidden,
                self.weights["classifier.dense.weight"],
                self.weights.get("classifier.dense.bias"),
            )
        )
        logits = _linear(
            pooled,
            self.weights["classifier.out_proj.weight"],
            self.weights.get("classifier.out_proj.bias"),
        )
        return _softmax(logits, axis=-1)


def _resolve_model_file(model_name_or_path: str, filename: str) -> Path | None:
    """Tìm file cấu hình/trọng số từ thư mục cục bộ hoặc tải qua Hugging Face Hub."""
    local_dir = Path(model_name_or_path)
    if local_dir.is_dir():
        candidate = local_dir / filename
        return candidate if candidate.is_file() else None

    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(repo_id=model_name_or_path, filename=filename))
    except Exception:
        return None


class _StandaloneXLMRobertaTokenizer:
    """Tokenizer độc lập dùng `tokenizers` hoặc `sentencepiece` (không kéo theo `transformers`/`torch`)."""

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 256,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        unk_token_id: int = 3,
    ) -> None:
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id
        self._hf_tokenizer: Any = None
        self._sp: Any = None

        tok_json = _resolve_model_file(model_name_or_path, "tokenizer.json")
        if tok_json is not None:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(tok_json))
            tok.enable_truncation(max_length=self.max_length)
            tok.enable_padding(pad_id=self.pad_token_id, pad_token="<pad>")
            self._hf_tokenizer = tok
            return

        sp_model = _resolve_model_file(model_name_or_path, "sentencepiece.bpe.model")
        if sp_model is not None:
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor()
            sp.Load(str(sp_model))
            self._sp = sp
            return

        raise FileNotFoundError(
            f"Không tìm thấy `tokenizer.json` hoặc `sentencepiece.bpe.model` cho `{model_name_or_path}`."
        )

    def _sp_id_to_fairseq(self, sp_id: int) -> int:
        if sp_id == 0:
            return self.unk_token_id
        if sp_id == 1:
            return self.bos_token_id
        if sp_id == 2:
            return self.eos_token_id
        return sp_id + 1

    def __call__(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int | None = None,
    ) -> dict[str, list[list[int]]]:
        limit = max_length or self.max_length

        if self._hf_tokenizer is not None:
            encodings = self._hf_tokenizer.encode_batch(texts)
            input_ids = [enc.ids[:limit] for enc in encodings]
            attention_mask = [enc.attention_mask[:limit] for enc in encodings]
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        assert self._sp is not None
        encoded_seqs: list[list[int]] = []
        inner_limit = max(0, limit - 2)
        for text in texts:
            sp_ids = self._sp.EncodeAsIds(text)
            mapped = [self._sp_id_to_fairseq(i) for i in sp_ids]
            if truncation:
                mapped = mapped[:inner_limit]
            encoded_seqs.append([self.bos_token_id] + mapped + [self.eos_token_id])

        max_seq_len = max((len(seq) for seq in encoded_seqs), default=0) if padding else 0
        input_ids_out: list[list[int]] = []
        attn_out: list[list[int]] = []
        for seq in encoded_seqs:
            pad_len = max(0, max_seq_len - len(seq)) if padding else 0
            input_ids_out.append(seq + [self.pad_token_id] * pad_len)
            attn_out.append([1] * len(seq) + [0] * pad_len)
        return {"input_ids": input_ids_out, "attention_mask": attn_out}


class CommentClassifier:
    """Giai đoạn 2: Phân loại cảm xúc comment tiếng Việt bằng ViSoBERT."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 256,
        normalize_teencode: bool = True,
        backend: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device_preference = device
        self.device = self._resolve_device(device)
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(16, int(max_length))
        self.normalize_teencode = normalize_teencode
        self.backend_preference = (backend or "auto").strip().lower()
        self.active_backend: str | None = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._numpy_model: _NumpyXLMRobertaClassifier | None = None
        self._id2label_norm: dict[int, str] = {}

    @staticmethod
    def _resolve_device(device: str) -> str:
        choice = (device or "cpu").strip().lower()
        if choice in {"auto", "cuda"}:
            try:
                import torch

                if choice == "auto":
                    return "cuda" if torch.cuda.is_available() else "cpu"
                if not torch.cuda.is_available():
                    print("[Cảnh báo] Không phát hiện CUDA khả dụng, chuyển về chạy trên CPU.")
                    return "cpu"
            except (ImportError, OSError):
                print("[Cảnh báo] Không nạp được PyTorch, chuyển về chạy trên CPU.")
                return "cpu"
        return choice

    @staticmethod
    def _normalize_raw_label(raw_label: str) -> str:
        upper = str(raw_label).strip().upper()
        if upper in {"POS", "POSITIVE", "LABEL_2"}:
            return "POS"
        if upper in {"NEG", "NEGATIVE", "LABEL_0"}:
            return "NEG"
        if upper in {"NEU", "NEUTRAL", "LABEL_1"}:
            return "NEU"
        if "POS" in upper:
            return "POS"
        if "NEG" in upper:
            return "NEG"
        return "NEU"

    def load(self) -> None:
        """Tải tokenizer và trọng số ViSoBERT (tự chuyển sang NumPy nếu PyTorch lỗi/bị chặn DLL)."""
        if self._tokenizer is not None and (self._model is not None or self._numpy_model is not None):
            return

        if self.backend_preference not in {"auto", "torch", "numpy"}:
            raise ValueError(
                f"Backend không hợp lệ: {self.backend_preference!r} (chọn 'auto', 'torch', hoặc 'numpy')."
            )

        if self.backend_preference in {"auto", "torch"}:
            try:
                import torch  # noqa: F401
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self._model.to(self.device)
                self._model.eval()
                self.active_backend = "torch"
                raw_id2label = getattr(self._model.config, "id2label", None) or {
                    0: "NEG",
                    1: "POS",
                    2: "NEU",
                }
                self._id2label_norm = {
                    int(idx): self._normalize_raw_label(label)
                    for idx, label in raw_id2label.items()
                }
                return
            except (ImportError, OSError) as exc:
                if self.backend_preference == "torch":
                    raise
                print(
                    f"[Thông báo] Không nạp được PyTorch ({exc}). "
                    "Tự động chuyển sang chế độ suy luận thuần NumPy + safetensors cho ViSoBERT."
                )

        self._numpy_model = _NumpyXLMRobertaClassifier(self.model_name)
        cfg = self._numpy_model.config
        self._tokenizer = _StandaloneXLMRobertaTokenizer(
            self.model_name,
            max_length=self.max_length,
            pad_token_id=int(cfg.get("pad_token_id", 1)),
            bos_token_id=int(cfg.get("bos_token_id", 0)),
            eos_token_id=int(cfg.get("eos_token_id", 2)),
            unk_token_id=int(cfg.get("unk_token_id", 3)),
        )
        self.active_backend = "numpy"
        self.device = "cpu"
        raw_id2label = cfg.get("id2label") or {
            "0": "NEG",
            "1": "POS",
            "2": "NEU",
        }
        self._id2label_norm = {
            int(idx): self._normalize_raw_label(label)
            for idx, label in raw_id2label.items()
        }

    def _predict_batch_probs(self, batch_texts: list[str]) -> list[list[float]]:
        """Chạy suy luận trên 1 batch text theo backend đang hoạt động ('numpy' hoặc 'torch')."""
        if self.active_backend == "numpy":
            assert self._numpy_model is not None
            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
            attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int64)
            return self._numpy_model.predict_probs(input_ids, attention_mask).tolist()

        import torch

        with torch.inference_mode():
            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            logits = self._model(**encoded).logits
            return torch.softmax(logits, dim=-1).cpu().tolist()

    def predict_batch(
        self, texts: Sequence[str], show_progress: bool = False
    ) -> list[CommentPrediction]:
        """Phân loại cảm xúc cho danh sách bình luận theo từng batch."""
        if not texts:
            return []

        self.load()

        preprocessed = [
            preprocess_comment(t, normalize_teencode=self.normalize_teencode)
            for t in texts
        ]
        results: list[CommentPrediction | None] = [None] * len(texts)
        valid_indices: list[int] = []

        for idx, (orig, cleaned) in enumerate(zip(texts, preprocessed)):
            if not cleaned:
                results[idx] = CommentPrediction(
                    original_text=str(orig),
                    cleaned_text="",
                    label="NEU",
                    label_vi=LABEL_VI_MAP["NEU"],
                    confidence=0.0,
                    scores={"POS": 0.0, "NEG": 0.0, "NEU": 1.0},
                )
            else:
                valid_indices.append(idx)

        batch_starts = range(0, len(valid_indices), self.batch_size)
        iterator = (
            tqdm(batch_starts, desc="Phân loại comment", unit="batch")
            if show_progress
            else batch_starts
        )

        for start in iterator:
            batch_idx = valid_indices[start : start + self.batch_size]
            batch_texts = [preprocessed[i] for i in batch_idx]
            probs = self._predict_batch_probs(batch_texts)

            for orig_idx, prob_row in zip(batch_idx, probs):
                score_map = {"POS": 0.0, "NEG": 0.0, "NEU": 0.0}
                for class_idx, p in enumerate(prob_row):
                    norm_label = self._id2label_norm.get(class_idx, "NEU")
                    score_map[norm_label] += float(p)

                best_label = max(score_map, key=score_map.get)  # type: ignore[arg-type]
                best_score = round(score_map[best_label], 4)
                rounded_scores = {k: round(v, 4) for k, v in score_map.items()}

                results[orig_idx] = CommentPrediction(
                    original_text=str(texts[orig_idx]),
                    cleaned_text=preprocessed[orig_idx],
                    label=best_label,
                    label_vi=LABEL_VI_MAP[best_label],
                    confidence=best_score,
                    scores=rounded_scores,
                )

        return [r for r in results if r is not None]

    def predict_one(self, text: str) -> CommentPrediction:
        """Phân loại cảm xúc cho một bình luận đơn lẻ."""
        return self.predict_batch([text], show_progress=False)[0]

    @staticmethod
    def summarize(
        predictions: Sequence[CommentPrediction], top_k: int = 3
    ) -> dict[str, Any]:
        """Tổng hợp thống kê cảm xúc, xu hướng chung và các comment tiêu biểu."""
        total = len(predictions)
        counts = {"POS": 0, "NEG": 0, "NEU": 0}
        if total == 0:
            return {
                "total_comments": 0,
                "counts": counts,
                "percentages": {"POS": 0.0, "NEG": 0.0, "NEU": 0.0},
                "avg_confidence": 0.0,
                "overall_tendency": "Không có dữ liệu",
                "top_examples": {"POS": [], "NEG": [], "NEU": []},
            }

        conf_sum = 0.0
        grouped: dict[str, list[CommentPrediction]] = {"POS": [], "NEG": [], "NEU": []}
        for pred in predictions:
            counts[pred.label] = counts.get(pred.label, 0) + 1
            conf_sum += pred.confidence
            grouped.setdefault(pred.label, []).append(pred)

        percentages = {
            label: round(count * 100.0 / total, 2)
            for label, count in counts.items()
        }

        if percentages["POS"] >= 35.0 and percentages["NEG"] >= 35.0:
            tendency = "Tranh cãi (Trái chiều)"
        else:
            dominant = max(counts, key=counts.get)  # type: ignore[arg-type]
            tendency = LABEL_VI_MAP[dominant]

        top_examples: dict[str, list[dict[str, Any]]] = {}
        for label in ("POS", "NEG", "NEU"):
            sorted_items = sorted(
                grouped.get(label, []),
                key=lambda x: x.confidence,
                reverse=True,
            )[: max(0, top_k)]
            top_examples[label] = [
                {
                    "comment": item.original_text,
                    "confidence": item.confidence,
                }
                for item in sorted_items
            ]

        return {
            "total_comments": total,
            "counts": counts,
            "percentages": percentages,
            "avg_confidence": round(conf_sum / total, 4),
            "overall_tendency": tendency,
            "top_examples": top_examples,
        }
