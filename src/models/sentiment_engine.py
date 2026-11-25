from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from tqdm import tqdm

from src.models.classical_ml import (
    DEFAULT_CACHE_DIR,
    EMOTION_LABELS,
    EMOTION_VI_MAP,
    BilingualLexiconScorer,
    ClassicalMLSuite,
)
from src.preprocessing.text_cleaner import PreprocessedComment

DEFAULT_VISOBERT_MODEL = "5CD-AI/Vietnamese-Sentiment-visobert"
DEFAULT_PHOBERT_SENTIMENT_MODEL = "wonrax/phobert-base-vietnamese-sentiment"
DEFAULT_PHOBERT_EMOTION_MODEL = "visolex/phobert-emotion"

DEFAULT_EN_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_EN_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

LABEL_VI_MAP: dict[str, str] = {
    "POS": "Tích cực",
    "NEG": "Tiêu cực",
    "NEU": "Trung tính",
}

RATING_5_VI_MAP: dict[int, str] = {
    1: "Rất tiêu cực",
    2: "Tiêu cực",
    3: "Trung tính",
    4: "Tích cực",
    5: "Rất tích cực",
}

_FEAR_CUES = (
    "sợ quá", "lo quá", "lo lắng", "hoang mang", "hãi", "run quá", "ớn lạnh",
    "scared", "terrified", "afraid", "creepy", "fear",
)
_ANGER_CUES = (
    "tức quá", "bực mình", "cay cú", "thù luôn", "phẫn nộ", "đấm", "điên tiết",
    "furious", "pissed off", "angry",
)
_DISGUST_CUES = (
    "mắc ói", "vô duyên", "kinh tởm", "tởm", "ghê tởm", "đơm đặt", "giả tạo", "lố bịch",
    "disgusting", "gross", "cringe", "pathetic",
)
_SADNESS_CUES = (
    "buồn quá", "khóc", "tội quá", "tội nghiệp", "thất vọng", "tụt năng lượng", "chán quá",
    "heartbroken", "depressing", "disappointed",
)
_JOY_CUES = (
    "vui quá", "hạnh phúc", "tuyệt vời", "mừng cho", "yêu thương", "thăng hoa", "dễ thương",
    "peak", "fire", "smooth", "incredible", "so cute", "freaking good",
)


@dataclass
class AnalyzedComment:
    """Kết quả phân tích đa tầng song ngữ (ML cổ điển + DL chuyên biệt + Hybrid + Thang điểm nhất quán)."""

    comment_id: str
    username: str
    nickname: str
    likes: int
    reply_count: int
    created_at: str
    post_url: str
    language: str  # "vi" | "en"
    original_text: str
    cleaned_text: str
    segmented_text: str
    ml_text: str
    is_duplicate: bool
    has_sarcasm_signal: bool
    needs_audit: bool
    # Nhãn & điểm hệ thống Hybrid chính
    polarity_label: str
    polarity_vi: str
    polarity_confidence: float
    polarity_score: float  # Thang [-1.0, +1.0] nhất quán với polarity_label
    score_5: int           # Thang [1..5] nhất quán với polarity_label (NEG:1-2, NEU:3, POS:4-5)
    score_10: float        # Thang [0.0..10.0]
    rating_level_vi: str
    polarity_probs: dict[str, float]
    # Cảm xúc 7 cung bậc
    emotion_label: str
    emotion_vi: str
    emotion_confidence: float
    emotion_probs: dict[str, float]
    # Dự đoán từng mô hình riêng lẻ để so sánh đối chiếu
    pred_nb: str
    pred_lr: str
    pred_svm: str
    pred_lexicon: str
    pred_visobert: str     # Chỉ chạy cho comment 'vi' (rỗng nếu 'en')
    pred_dl_primary: str   # PhoBERT (cho 'vi') / Twitter-RoBERTa (cho 'en')
    emotion_ml_label: str
    model_probs: dict[str, dict[str, float]] = field(default_factory=dict)
    is_reply: bool = False
    parent_comment_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_dual_scale(
    probs: dict[str, float],
    polarity_label: str,
    emotion_label: str,
    emotion_conf: float,
) -> tuple[float, int, float, str]:
    """Quy đổi xác suất thành Thang [-1,+1], [1..5] và [0..10] nhất quán tuyệt đối với polarity_label."""
    p_pos = float(probs.get("POS", 0.0))
    p_neg = float(probs.get("NEG", 0.0))
    p_neu = float(probs.get("NEU", 0.0))

    nudge = 0.0
    if emotion_label in {"Anger", "Disgust"} and emotion_conf >= 0.5:
        nudge = -0.08 * emotion_conf
    elif emotion_label in {"Sadness", "Fear"} and emotion_conf >= 0.5:
        nudge = -0.04 * emotion_conf
    elif emotion_label == "Enjoyment" and emotion_conf >= 0.5:
        nudge = 0.06 * emotion_conf

    raw = p_pos - p_neg + nudge

    if polarity_label == "NEG":
        polarity_score = float(np.clip(min(raw, -0.21), -1.0, -0.21))
        score_5 = 1 if polarity_score <= -0.60 else 2
    elif polarity_label == "POS":
        polarity_score = float(np.clip(max(raw, 0.21), 0.21, 1.0))
        score_5 = 5 if polarity_score >= 0.60 else 4
    else:
        polarity_score = float(np.clip(raw * (1.0 - 0.5 * p_neu), -0.19, 0.19))
        score_5 = 3

    polarity_score = round(polarity_score, 4)
    score_10 = round((polarity_score + 1.0) * 5.0, 2)
    return polarity_score, score_5, score_10, RATING_5_VI_MAP[score_5]


class _HFSequenceClassifier:
    """Wrapper chuẩn cho Transformer sequence classification chạy trên CPU (bảo vệ 4GB VRAM)."""

    def __init__(
        self,
        model_name: str,
        label_normalizer: Any,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.label_normalizer = label_normalizer
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(16, int(max_length))
        self._tokenizer: Any = None
        self._model: Any = None
        self._id2label: dict[int, str] = {}

    def load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        raw_map = getattr(self._model.config, "id2label", None) or {}
        self._id2label = {
            int(idx): self.label_normalizer(lbl, int(idx))
            for idx, lbl in raw_map.items()
        }

    def predict_probs(
        self, texts: Sequence[str], target_labels: Sequence[str], show_progress: bool = False
    ) -> list[dict[str, float]]:
        if not texts:
            return []
        self.load()
        import torch

        results: list[dict[str, float]] = []
        starts = range(0, len(texts), self.batch_size)
        iterator = (
            tqdm(starts, desc=f"Model {self.model_name.split('/')[-1]}", unit="batch")
            if show_progress
            else starts
        )
        with torch.inference_mode():
            for start in iterator:
                batch = [str(t) for t in texts[start : start + self.batch_size]]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                logits = self._model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().tolist()
                for row in probs:
                    score_map = {lbl: 0.0 for lbl in target_labels}
                    for idx, p in enumerate(row):
                        norm_lbl = self._id2label.get(idx)
                        if norm_lbl in score_map:
                            score_map[norm_lbl] += float(p)
                    results.append({k: round(v, 4) for k, v in score_map.items()})
        return results


def _norm_vi_polarity_label(raw_label: str, idx: int) -> str:
    up = str(raw_label).strip().upper()
    if up in {"POS", "POSITIVE", "LABEL_1"}:
        return "POS"
    if up in {"NEG", "NEGATIVE", "LABEL_0"}:
        return "NEG"
    if up in {"NEU", "NEUTRAL", "LABEL_2"}:
        return "NEU"
    if "POS" in up:
        return "POS"
    if "NEG" in up:
        return "NEG"
    if "NEU" in up:
        return "NEU"
    return {0: "NEG", 1: "POS", 2: "NEU"}.get(idx, "NEU")


def _norm_en_polarity_label(raw_label: str, idx: int) -> str:
    """Chuẩn hóa nhãn cho cardiffnlp/twitter-roberta-base-sentiment-latest (0=NEG, 1=NEU, 2=POS)."""
    up = str(raw_label).strip().upper()
    if "POS" in up:
        return "POS"
    if "NEG" in up:
        return "NEG"
    if "NEU" in up:
        return "NEU"
    return {0: "NEG", 1: "NEU", 2: "POS"}.get(idx, "NEU")


def _norm_emotion_label(raw_label: str, idx: int) -> str:
    """Chuẩn hóa nhãn cảm xúc cho cả visolex/phobert-emotion và j-hartmann/emotion-english-distilroberta-base."""
    low = str(raw_label).strip().lower()
    mapping = {
        "anger": "Anger",
        "angry": "Anger",
        "disgust": "Disgust",
        "enjoyment": "Enjoyment",
        "joy": "Enjoyment",
        "happy": "Enjoyment",
        "fear": "Fear",
        "sadness": "Sadness",
        "sad": "Sadness",
        "surprise": "Surprise",
        "neutral": "Other",
        "other": "Other",
    }
    for key, val in mapping.items():
        if key in low:
            return val
    return "Other"


class CommentAnalyzer:
    """Giai đoạn 3: Hệ thống phân tích đa tầng song ngữ (Tiếng Việt & Tiếng Anh)."""

    def __init__(
        self,
        visobert_model: str = DEFAULT_VISOBERT_MODEL,
        phobert_sentiment_model: str = DEFAULT_PHOBERT_SENTIMENT_MODEL,
        phobert_emotion_model: str = DEFAULT_PHOBERT_EMOTION_MODEL,
        en_sentiment_model: str = DEFAULT_EN_SENTIMENT_MODEL,
        en_emotion_model: str = DEFAULT_EN_EMOTION_MODEL,
        classical_cache_dir: str | Path = DEFAULT_CACHE_DIR,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 256,
    ) -> None:
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(16, int(max_length))

        self.classical_suite = ClassicalMLSuite(cache_dir=classical_cache_dir)
        self.lexicon_scorer = BilingualLexiconScorer()

        # Nhánh Tiếng Việt (lazy load)
        self.vi_visobert = _HFSequenceClassifier(
            model_name=visobert_model,
            label_normalizer=_norm_vi_polarity_label,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        self.vi_phobert_sentiment = _HFSequenceClassifier(
            model_name=phobert_sentiment_model,
            label_normalizer=_norm_vi_polarity_label,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        self.vi_phobert_emotion = _HFSequenceClassifier(
            model_name=phobert_emotion_model,
            label_normalizer=_norm_emotion_label,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )

        # Nhánh Tiếng Anh (lazy load)
        self.en_roberta_sentiment = _HFSequenceClassifier(
            model_name=en_sentiment_model,
            label_normalizer=_norm_en_polarity_label,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        self.en_roberta_emotion = _HFSequenceClassifier(
            model_name=en_emotion_model,
            label_normalizer=_norm_emotion_label,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )

        self.last_model_timings: dict[str, float] = {}

    @staticmethod
    def _apply_emotion_cues(
        emo_probs: dict[str, float],
        cleaned_text: str,
    ) -> dict[str, float]:
        low = cleaned_text.lower()
        boosted = dict(emo_probs)
        if any(cue in low for cue in _ANGER_CUES):
            boosted["Anger"] = boosted.get("Anger", 0.0) + 0.25
        if any(cue in low for cue in _DISGUST_CUES):
            boosted["Disgust"] = boosted.get("Disgust", 0.0) + 0.25
        if any(cue in low for cue in _FEAR_CUES):
            boosted["Fear"] = boosted.get("Fear", 0.0) + 0.30
        if any(cue in low for cue in _SADNESS_CUES):
            boosted["Sadness"] = boosted.get("Sadness", 0.0) + 0.20
        if any(cue in low for cue in _JOY_CUES):
            boosted["Enjoyment"] = boosted.get("Enjoyment", 0.0) + 0.15

        total = sum(boosted.values())
        if total > 0:
            boosted = {k: round(v / total, 4) for k, v in boosted.items()}
        return boosted

    def analyze_comments(
        self,
        preprocessed: Sequence[PreprocessedComment],
        metadata_rows: Sequence[dict[str, Any]] | None = None,
        show_progress: bool = True,
    ) -> tuple[list[AnalyzedComment], dict[str, float]]:
        """Phân tích toàn bộ comment hợp lệ (tự định tuyến VI/EN qua mô hình đúng ngôn ngữ)."""
        valid_indices = [idx for idx, item in enumerate(preprocessed) if item.is_valid]
        if not valid_indices:
            self.last_model_timings = {}
            return [], {}

        valid_items = [preprocessed[i] for i in valid_indices]
        n_valid = len(valid_items)
        cleaned_texts = [it.cleaned_text for it in valid_items]
        segmented_texts = [it.segmented_text or it.cleaned_text for it in valid_items]
        languages = ["en" if it.language == "en" else "vi" for it in valid_items]

        vi_positions = [i for i, l in enumerate(languages) if l == "vi"]
        en_positions = [i for i, l in enumerate(languages) if l == "en"]

        timings: dict[str, float] = {}

        # 1. Nạp và chạy ML cổ điển song ngữ (NB, LR, SVM + Emotion LR)
        t0 = time.perf_counter()
        if vi_positions:
            self.classical_suite.ensure_loaded("vi")
        if en_positions:
            self.classical_suite.ensure_loaded("en")
        timings["classical_ml_setup_sec"] = round(time.perf_counter() - t0, 3)

        t1 = time.perf_counter()
        classical_pol = self.classical_suite.predict_polarity_all(
            segmented_texts, languages=languages
        )
        timings["classical_ml_polarity_sec"] = round(time.perf_counter() - t1, 3)

        t2 = time.perf_counter()
        classical_emo = self.classical_suite.predict_emotion(
            segmented_texts, languages=languages
        )
        timings["classical_ml_emotion_sec"] = round(time.perf_counter() - t2, 3)

        # 2. Chạy Bilingual Lexicon + Rules
        t3 = time.perf_counter()
        lex_results = self.lexicon_scorer.predict_batch(
            segmented_texts, cleaned_texts, languages=languages
        )
        timings["lexicon_rules_sec"] = round(time.perf_counter() - t3, 3)

        # 3. Chạy Deep Learning chuyên biệt theo ngôn ngữ (lazy-load)
        viso_probs_all: list[dict[str, float] | None] = [None] * n_valid
        dl_primary_probs_all: list[dict[str, float] | None] = [None] * n_valid
        dl_emo_probs_all: list[dict[str, float] | None] = [None] * n_valid

        if vi_positions:
            vi_clean = [cleaned_texts[i] for i in vi_positions]
            vi_seg = [segmented_texts[i] for i in vi_positions]

            t_v = time.perf_counter()
            vi_viso = self.vi_visobert.predict_probs(
                vi_clean, target_labels=("POS", "NEG", "NEU"), show_progress=show_progress
            )
            timings["visobert_sec"] = round(time.perf_counter() - t_v, 3)

            t_p = time.perf_counter()
            vi_pho = self.vi_phobert_sentiment.predict_probs(
                vi_seg, target_labels=("POS", "NEG", "NEU"), show_progress=show_progress
            )
            timings["phobert_sentiment_sec"] = round(time.perf_counter() - t_p, 3)

            t_pe = time.perf_counter()
            vi_emo = self.vi_phobert_emotion.predict_probs(
                vi_seg, target_labels=EMOTION_LABELS, show_progress=show_progress
            )
            timings["phobert_emotion_sec"] = round(time.perf_counter() - t_pe, 3)

            for pos, vp, pp, ep in zip(vi_positions, vi_viso, vi_pho, vi_emo):
                viso_probs_all[pos] = vp
                dl_primary_probs_all[pos] = pp
                dl_emo_probs_all[pos] = ep

        if en_positions:
            en_clean = [cleaned_texts[i] for i in en_positions]

            t_er = time.perf_counter()
            en_rob = self.en_roberta_sentiment.predict_probs(
                en_clean, target_labels=("POS", "NEG", "NEU"), show_progress=show_progress
            )
            timings["en_roberta_sentiment_sec"] = round(time.perf_counter() - t_er, 3)

            t_ee = time.perf_counter()
            en_emo = self.en_roberta_emotion.predict_probs(
                en_clean, target_labels=EMOTION_LABELS, show_progress=show_progress
            )
            timings["en_roberta_emotion_sec"] = round(time.perf_counter() - t_ee, 3)

            for pos, rp, ep in zip(en_positions, en_rob, en_emo):
                dl_primary_probs_all[pos] = rp
                dl_emo_probs_all[pos] = ep

        # 4. Tổ hợp Hybrid Ensemble + Thang điểm kép nhất quán
        analyzed: list[AnalyzedComment] = []
        for pos_idx, orig_idx in enumerate(valid_indices):
            item = preprocessed[orig_idx]
            lang = languages[pos_idx]
            meta = metadata_rows[orig_idx] if metadata_rows and orig_idx < len(metadata_rows) else {}

            nb_res = classical_pol["tfidf_nb"][pos_idx]
            lr_res = classical_pol["tfidf_lr"][pos_idx]
            svm_res = classical_pol["tfidf_svm"][pos_idx]
            lex_res = lex_results[pos_idx]

            dl_probs = dl_primary_probs_all[pos_idx] or {"POS": 0.33, "NEG": 0.33, "NEU": 0.34}
            dl_lbl = max(dl_probs, key=dl_probs.get)  # type: ignore[arg-type]

            viso_probs = viso_probs_all[pos_idx]
            viso_lbl = max(viso_probs, key=viso_probs.get) if viso_probs else ""  # type: ignore[arg-type]

            raw_emo_probs = dl_emo_probs_all[pos_idx] or {lbl: (1.0 if lbl == "Other" else 0.0) for lbl in EMOTION_LABELS}
            emo_probs = self._apply_emotion_cues(raw_emo_probs, item.cleaned_text)
            emo_label = max(emo_probs, key=emo_probs.get)  # type: ignore[arg-type]
            emo_conf = round(float(emo_probs[emo_label]), 4)
            emo_ml_label = classical_emo[pos_idx]["label"]

            hybrid_probs: dict[str, float] = {}
            if lang == "vi" and viso_probs is not None:
                for lbl in ("POS", "NEG", "NEU"):
                    hybrid_probs[lbl] = (
                        0.40 * float(dl_probs.get(lbl, 0.0))
                        + 0.20 * float(viso_probs.get(lbl, 0.0))
                        + 0.25 * float(svm_res["scores"].get(lbl, 0.0))
                        + 0.15 * float(lex_res["scores"].get(lbl, 0.0))
                    )
            else:
                # Nhánh Tiếng Anh: 0.60 Twitter-RoBERTa + 0.25 SVM(EN) + 0.15 Lexicon(EN)
                for lbl in ("POS", "NEG", "NEU"):
                    hybrid_probs[lbl] = (
                        0.60 * float(dl_probs.get(lbl, 0.0))
                        + 0.25 * float(svm_res["scores"].get(lbl, 0.0))
                        + 0.15 * float(lex_res["scores"].get(lbl, 0.0))
                    )

            # Cơ chế nhất quán Emotion-Polarity
            if (
                emo_label in {"Anger", "Disgust", "Fear"}
                and emo_conf >= 0.55
                and float(lex_res["scores"].get("NEG", 0.0)) >= 0.50
            ):
                hybrid_probs["NEG"] += 0.25
                hybrid_probs["POS"] *= 0.65

            total_h = sum(hybrid_probs.values()) or 1.0
            hybrid_probs = {k: round(v / total_h, 4) for k, v in hybrid_probs.items()}

            pol_label = max(hybrid_probs, key=hybrid_probs.get)  # type: ignore[arg-type]
            pol_conf = round(float(hybrid_probs[pol_label]), 4)

            polarity_score, score_5, score_10, rating_vi = compute_dual_scale(
                hybrid_probs,
                polarity_label=pol_label,
                emotion_label=emo_label,
                emotion_conf=emo_conf,
            )

            needs_audit = (
                (bool(viso_lbl) and dl_lbl != viso_lbl)
                or (pol_conf < 0.65)
                or item.has_sarcasm_signal
            )

            model_probs_entry: dict[str, dict[str, float]] = {
                "nb": nb_res["scores"],
                "lr": lr_res["scores"],
                "svm": svm_res["scores"],
                "lexicon": lex_res["scores"],
                "dl_primary": dl_probs,
            }
            if viso_probs is not None:
                model_probs_entry["visobert"] = viso_probs

            raw_is_reply = meta.get("is_reply", False)
            if isinstance(raw_is_reply, str):
                is_reply_val = raw_is_reply.strip().lower() in {"true", "1", "yes", "y"}
            else:
                try:
                    is_reply_val = bool(raw_is_reply) and not pd.isna(raw_is_reply)
                except Exception:
                    is_reply_val = bool(raw_is_reply)

            analyzed.append(
                AnalyzedComment(
                    comment_id=str(meta.get("comment_id", "")),
                    username=str(meta.get("username", "")),
                    nickname=str(meta.get("nickname", "")),
                    likes=int(meta.get("likes", 0) or 0),
                    reply_count=int(meta.get("reply_count", 0) or 0),
                    created_at=str(meta.get("created_at", "")),
                    post_url=str(meta.get("post_url", "")),
                    language=lang,
                    original_text=item.original_text,
                    cleaned_text=item.cleaned_text,
                    segmented_text=item.segmented_text,
                    ml_text=item.ml_text,
                    is_duplicate=item.is_duplicate,
                    has_sarcasm_signal=item.has_sarcasm_signal,
                    needs_audit=needs_audit,
                    polarity_label=pol_label,
                    polarity_vi=LABEL_VI_MAP[pol_label],
                    polarity_confidence=pol_conf,
                    polarity_score=polarity_score,
                    score_5=score_5,
                    score_10=score_10,
                    rating_level_vi=rating_vi,
                    polarity_probs=hybrid_probs,
                    emotion_label=emo_label,
                    emotion_vi=EMOTION_VI_MAP.get(emo_label, emo_label),
                    emotion_confidence=emo_conf,
                    emotion_probs=emo_probs,
                    pred_nb=nb_res["label"],
                    pred_lr=lr_res["label"],
                    pred_svm=svm_res["label"],
                    pred_lexicon=lex_res["label"],
                    pred_visobert=viso_lbl,
                    pred_dl_primary=dl_lbl,
                    emotion_ml_label=emo_ml_label,
                    model_probs=model_probs_entry,
                    is_reply=is_reply_val,
                    parent_comment_id=str(meta.get("parent_comment_id", "") or "").strip(),
                )
            )

        self.last_model_timings = timings
        return analyzed, timings

    def summarize(
        self,
        analyzed: Sequence[AnalyzedComment],
        deduplicate_stats: bool = False,
        top_k: int = 3,
    ) -> dict[str, Any]:
        items = (
            [a for a in analyzed if not a.is_duplicate]
            if deduplicate_stats
            else list(analyzed)
        )
        total = len(items)
        pol_counts = {"POS": 0, "NEG": 0, "NEU": 0}
        emo_counts = {lbl: 0 for lbl in EMOTION_LABELS}
        score_5_counts = {str(i): 0 for i in range(1, 6)}
        lang_counts = {"vi": 0, "en": 0}

        if total == 0:
            return {
                "total_valid_comments": 0,
                "language_counts": lang_counts,
                "polarity_counts": pol_counts,
                "polarity_percentages": {"POS": 0.0, "NEG": 0.0, "NEU": 0.0},
                "emotion_counts": emo_counts,
                "emotion_percentages": {lbl: 0.0 for lbl in EMOTION_LABELS},
                "score_5_counts": score_5_counts,
                "avg_polarity_score": 0.0,
                "avg_score_5": 0.0,
                "avg_score_10": 0.0,
                "avg_confidence": 0.0,
                "overall_tendency": "Không có dữ liệu",
                "sarcasm_flagged_count": 0,
                "classical_ml_benchmark": self.classical_suite.benchmark_metrics,
                "model_timings_sec": self.last_model_timings,
                "top_examples": {"POS": [], "NEG": [], "NEU": []},
            }

        conf_sum = 0.0
        pol_score_sum = 0.0
        s5_sum = 0
        s10_sum = 0.0
        sarcasm_count = 0
        grouped: dict[str, list[AnalyzedComment]] = {"POS": [], "NEG": [], "NEU": []}

        for a in items:
            lang_counts[a.language] = lang_counts.get(a.language, 0) + 1
            pol_counts[a.polarity_label] = pol_counts.get(a.polarity_label, 0) + 1
            emo_counts[a.emotion_label] = emo_counts.get(a.emotion_label, 0) + 1
            score_5_counts[str(a.score_5)] = score_5_counts.get(str(a.score_5), 0) + 1
            conf_sum += a.polarity_confidence
            pol_score_sum += a.polarity_score
            s5_sum += a.score_5
            s10_sum += a.score_10
            if a.has_sarcasm_signal:
                sarcasm_count += 1
            grouped.setdefault(a.polarity_label, []).append(a)

        pol_pcts = {k: round(v * 100.0 / total, 2) for k, v in pol_counts.items()}
        emo_pcts = {k: round(v * 100.0 / total, 2) for k, v in emo_counts.items()}

        if pol_pcts["POS"] >= 35.0 and pol_pcts["NEG"] >= 35.0:
            tendency = "Tranh cãi (Trái chiều)"
        else:
            dominant = max(pol_counts, key=pol_counts.get)  # type: ignore[arg-type]
            tendency = LABEL_VI_MAP[dominant]

        top_examples: dict[str, list[dict[str, Any]]] = {}
        for lbl in ("POS", "NEG", "NEU"):
            sorted_items = sorted(
                grouped.get(lbl, []),
                key=lambda x: (x.likes, x.polarity_confidence),
                reverse=True,
            )[: max(0, top_k)]
            top_examples[lbl] = [
                {
                    "comment": it.original_text,
                    "language": it.language,
                    "likes": it.likes,
                    "score_5": it.score_5,
                    "polarity_score": it.polarity_score,
                    "emotion": it.emotion_vi,
                    "confidence": it.polarity_confidence,
                }
                for it in sorted_items
            ]

        return {
            "total_valid_comments": total,
            "language_counts": lang_counts,
            "polarity_counts": pol_counts,
            "polarity_percentages": pol_pcts,
            "emotion_counts": emo_counts,
            "emotion_percentages": emo_pcts,
            "score_5_counts": score_5_counts,
            "avg_polarity_score": round(pol_score_sum / total, 4),
            "avg_score_5": round(s5_sum / total, 2),
            "avg_score_10": round(s10_sum / total, 2),
            "avg_confidence": round(conf_sum / total, 4),
            "overall_tendency": tendency,
            "sarcasm_flagged_count": sarcasm_count,
            "classical_ml_benchmark": self.classical_suite.benchmark_metrics,
            "model_timings_sec": self.last_model_timings,
            "top_examples": top_examples,
        }
