from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import requests
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from src.preprocessing.text_cleaner import clean_comment_text

ML_BUNDLE_SCHEMA_VERSION = 3
POLARITY_LABELS = ("POS", "NEU", "NEG")
POLARITY_MODEL_KEYS = (
    "tfidf_nb",
    "tfidf_cnb",
    "tfidf_lr",
    "tfidf_ridge",
    "tfidf_sgd",
    "tfidf_svm",
    "tfidf_nbsvm",
    "tfidf_rbf_svm",
    "tfidf_rf",
    "tfidf_tree",
    "ml_ensemble",
)
EMOTION_MODEL_KEYS = (
    "emo_cnb",
    "emo_lr",
    "emo_ridge",
    "emo_svm",
    "emo_nbsvm",
    "emo_sgd",
    "emo_ensemble",
)

CORE_POLARITY_MODEL_KEYS = (
    "tfidf_cnb",
    "tfidf_lr",
    "tfidf_svm",
    "tfidf_rf",
    "tfidf_tree",
    "ml_ensemble",
)

CORE_EMOTION_MODEL_KEYS = (
    "emo_cnb",
    "emo_lr",
    "emo_svm",
    "emo_ensemble",
)

POLARITY_MODEL_META: dict[str, dict[str, str]] = {
    "tfidf_nb": {
        "display_name": "TF-IDF + Multinomial NB",
        "short_name": "Multinomial NB",
        "family": "Classical ML (Generative)",
        "family_vi": "Xác suất sinh (Naive Bayes)",
        "family_color": "#0ea5e9",
        "desc_vi": "Đếm xác suất từ xuất hiện theo từng lớp dựa trên định lý Bayes.",
    },
    "tfidf_cnb": {
        "display_name": "TF-IDF + Complement NB",
        "short_name": "Complement NB",
        "family": "Classical ML (Generative)",
        "family_vi": "Xác suất sinh (Naive Bayes)",
        "family_color": "#0ea5e9",
        "desc_vi": "Ước lượng xác suất phần bù giúp xử lý tốt tập bình luận mất cân bằng lớp.",
    },
    "tfidf_lr": {
        "display_name": "TF-IDF + Logistic Regression",
        "short_name": "Logistic Regression",
        "family": "Classical ML (Linear)",
        "family_vi": "Tuyến tính (Linear)",
        "family_color": "#6366f1",
        "desc_vi": "Học trọng số tuyến tính cho từng từ/n-gram và xuất xác suất qua Softmax.",
    },
    "tfidf_ridge": {
        "display_name": "TF-IDF + Ridge Classifier",
        "short_name": "Ridge Classifier",
        "family": "Classical ML (Linear)",
        "family_vi": "Tuyến tính (Linear)",
        "family_color": "#6366f1",
        "desc_vi": "Phân loại tuyến tính với chuẩn hóa L2 giúp tránh quá khớp trên từ điển lớn.",
    },
    "tfidf_sgd": {
        "display_name": "TF-IDF + SGD Classifier",
        "short_name": "SGD Classifier",
        "family": "Classical ML (Linear)",
        "family_vi": "Tuyến tính (Linear)",
        "family_color": "#6366f1",
        "desc_vi": "Tối ưu tuyến tính bằng hạ gradient ngẫu nhiên (log_loss / modified_huber).",
    },
    "tfidf_svm": {
        "display_name": "TF-IDF + Linear SVM",
        "short_name": "Linear SVM",
        "family": "Classical ML (Margin / SVM)",
        "family_vi": "Biên cực đại (SVM)",
        "family_color": "#8b5cf6",
        "desc_vi": "Tìm siêu phẳng có biên phân cách lớn nhất trong không gian TF-IDF nhiều chiều.",
    },
    "tfidf_nbsvm": {
        "display_name": "TF-IDF + NBSVM (NB-Weighted SVM)",
        "short_name": "NBSVM",
        "family": "Classical ML (Margin / SVM)",
        "family_vi": "Biên cực đại (SVM)",
        "family_color": "#8b5cf6",
        "desc_vi": "Nhân đặc trưng với log-count ratio Naive Bayes trước khi đưa vào LinearSVC.",
    },
    "tfidf_rbf_svm": {
        "display_name": "SVD + RBF Kernel SVM",
        "short_name": "RBF Kernel SVM",
        "family": "Classical ML (Margin / SVM)",
        "family_vi": "Biên cực đại (SVM)",
        "family_color": "#8b5cf6",
        "desc_vi": "SVM hạt nhân phi tuyến RBF trên không gian nén TruncatedSVD(96) + Social.",
    },
    "tfidf_rf": {
        "display_name": "SVD+Social + Random Forest",
        "short_name": "Random Forest",
        "family": "Classical ML (Tree Bagging)",
        "family_vi": "Cây quyết định (Bagging)",
        "family_color": "#14b8a6",
        "desc_vi": "Bỏ phiếu từ 150 cây quyết định độc lập trên đặc trưng SVD + 15 tín hiệu MXH.",
    },
    "tfidf_tree": {
        "display_name": "SVD+Social + HistGradientBoosting",
        "short_name": "HistGradientBoosting",
        "family": "Classical ML (Tree Boosting)",
        "family_vi": "Cây quyết định (Boosting)",
        "family_color": "#f59e0b",
        "desc_vi": "Chuỗi cây tăng cường tuần tự sửa lỗi của cây trước trên đặc trưng SVD + MXH.",
    },
    "ml_ensemble": {
        "display_name": "Classical ML Meta-Ensemble",
        "short_name": "ML Meta-Ensemble",
        "family": "Classical ML Ensemble",
        "family_vi": "Tổ hợp ML (Meta-Ensemble)",
        "family_color": "#2563eb",
        "desc_vi": "Meta-learner kết hợp xác suất OOF từ nhóm mô hình ML cổ điển mạnh nhất.",
    },
}

EMOTION_MODEL_META: dict[str, dict[str, str]] = {
    "emo_cnb": {
        "display_name": "Complement NB (Emotion)",
        "short_name": "Complement NB",
        "family": "Classical ML (Generative)",
        "family_vi": "Xác suất sinh (Naive Bayes)",
        "family_color": "#0ea5e9",
        "desc_vi": "Nhận diện 7 cung bậc cảm xúc bằng phân phối phần bù trên đặc trưng TF-IDF.",
    },
    "emo_lr": {
        "display_name": "Logistic Regression (Emotion)",
        "short_name": "Logistic Regression",
        "family": "Classical ML (Linear)",
        "family_vi": "Tuyến tính (Linear)",
        "family_color": "#6366f1",
        "desc_vi": "Phân loại đa lớp 7 cảm xúc bằng trọng số tuyến tính và Softmax.",
    },
    "emo_ridge": {
        "display_name": "Ridge Classifier (Emotion)",
        "short_name": "Ridge Classifier",
        "family": "Classical ML (Linear)",
        "family_vi": "Tuyến tính (Linear)",
        "family_color": "#6366f1",
        "desc_vi": "Phân loại tuyến tính chuẩn hóa L2 cho 7 cung bậc cảm xúc.",
    },
    "emo_svm": {
        "display_name": "Linear SVM (Emotion)",
        "short_name": "Linear SVM",
        "family": "Classical ML (Margin / SVM)",
        "family_vi": "Biên cực đại (SVM)",
        "family_color": "#8b5cf6",
        "desc_vi": "Phân tách 7 cung bậc cảm xúc bằng siêu phẳng biên cực đại đã hiệu chỉnh xác suất.",
    },
    "emo_nbsvm": {
        "display_name": "NBSVM (Emotion)",
        "short_name": "NBSVM",
        "family": "Classical ML (Margin / SVM)",
        "family_vi": "Biên cực đại (SVM)",
        "family_color": "#8b5cf6",
        "desc_vi": "Kết hợp trọng số Naive Bayes và LinearSVC cho bài toán 7 cảm xúc.",
    },
    "emo_sgd": {
        "display_name": "SGD Classifier (Emotion)",
        "short_name": "SGD Classifier",
        "family": "Classical ML (Linear)",
        "family_vi": "Tuyến tính (Linear)",
        "family_color": "#6366f1",
        "desc_vi": "Huấn luyện tuyến tính theo SGD cho 7 cung bậc cảm xúc.",
    },
    "emo_ensemble": {
        "display_name": "ML Emotion Meta-Ensemble",
        "short_name": "ML Emotion Ensemble",
        "family": "Classical ML Ensemble",
        "family_vi": "Tổ hợp ML (Meta-Ensemble)",
        "family_color": "#2563eb",
        "desc_vi": "Tổng hợp dự đoán xác suất từ các mô hình ML Emotion mạnh nhất.",
    },
}

DEFAULT_CACHE_DIR = Path("models") / "classical_ml"

EMOTION_LABELS = (
    "Anger",
    "Sadness",
    "Fear",
    "Enjoyment",
    "Disgust",
    "Surprise",
    "Other",
)

EMOTION_VI_MAP: dict[str, str] = {
    "Anger": "Tức giận",
    "Sadness": "Buồn bã",
    "Fear": "Lo lắng / Sợ hãi",
    "Enjoyment": "Vui vẻ / Hài lòng",
    "Disgust": "Chán ghét / Khó chịu",
    "Surprise": "Ngạc nhiên",
    "Other": "Trung tính / Khác",
}

# --- Từ điển sắc thái Tiếng Việt ---
_VI_POSITIVE_LEXICON: dict[str, float] = {
    "tử_tế": 1.6, "tử tế": 1.6, "duyên_dáng": 1.5, "duyên dáng": 1.5,
    "có_duyên": 1.4, "có duyên": 1.4, "tuyệt_vời": 1.8, "tuyệt vời": 1.8,
    "xuất_sắc": 1.8, "xuất sắc": 1.8, "đỉnh": 1.4, "hay": 1.1, "giỏi": 1.3,
    "đẹp_trai": 1.4, "đẹp trai": 1.4, "xinh_đẹp": 1.4, "dễ_thương": 1.4,
    "dễ thương": 1.4, "thương": 1.1, "yêu_thương": 1.5, "chung_thuỷ": 1.5,
    "chung_thủy": 1.5, "thăng_hoa": 1.4, "ủng_hộ": 1.3, "ủng hộ": 1.3,
    "mừng": 1.2, "vui": 1.2, "vui_vẻ": 1.4, "hài_lòng": 1.4, "thích": 1.2,
    "yêu": 1.3, "đồng_hành": 1.1, "nâng_đỡ": 1.1, "chân_thành": 1.3,
    "tốt": 1.1, "ok": 0.7, "ổn": 0.7, "xịn": 1.3, "chuẩn": 1.0,
    "cảm_ơn": 1.1, "hạnh_phúc": 1.5, "ấm_áp": 1.3, "chất_lượng": 1.2,
    "❤️": 1.2, "♥️": 1.2, "🥰": 1.2, "😍": 1.3, "👏": 1.0, "👍": 1.0, "🥹": 0.6,
}

_VI_NEGATIVE_LEXICON: dict[str, float] = {
    "ghét": 1.7, "thù": 1.9, "vô_duyên": 1.8, "vô duyên": 1.8,
    "mắc_ói": 2.0, "mắc ói": 2.0, "buồn_nôn": 1.8, "cay_cú": 1.7,
    "cay cú": 1.7, "khó_chịu": 1.5, "khó chịu": 1.5, "chán": 1.4,
    "oải": 1.4, "tụt_năng_lượng": 1.6, "đơm_đặt": 1.7, "đơm đặt": 1.7,
    "tự_ti": 1.3, "ganh_tị": 1.6, "ganh tị": 1.6, "trù": 1.7, "khinh": 1.8,
    "lố": 1.4, "lố_bịch": 1.8, "giả_tạo": 1.7, "giả tạo": 1.7, "tệ": 1.6,
    "dở": 1.5, "tởm": 1.9, "phốt": 1.4, "drama": 0.9, "tào_lao": 1.5,
    "tào lao": 1.5, "nặng_nề": 1.3, "cảm_lạnh": 1.4, "cảm lạnh": 1.4,
    "tội_nghiệp": 1.0, "thất_vọng": 1.6, "thất vọng": 1.6, "bực": 1.5,
    "tức": 1.6, "tức_giận": 1.7, "bức_xúc": 1.6, "sợ": 1.2, "lo_lắng": 1.2,
    "buồn": 1.3, "khóc": 1.1, "đau_lòng": 1.5, "lừa_đảo": 1.9, "xấu": 1.4,
    "kém": 1.5, "rác": 1.7, "nhạt": 1.2, "xàm": 1.5, "đấm": 1.3,
    "chấn_lột": 1.4, "vcl": 1.1, "vl": 0.8, "mắc_cười": 0.3,
    "🙄": 1.2, "🤢": 1.7, "🤮": 1.8, "😡": 1.7, "😠": 1.6,
    "🤬": 1.8, "👎": 1.4, "🤡": 1.0, "😢": 1.0, "😭": 1.0,
}

_VI_NEGATION_WORDS = {"không", "chẳng", "chả", "đéo", "chưa", "hổng", "đừng", "chớ", "hông", "hem"}
_VI_ESCALATION_MARKERS = ("chứ không", "chứ ko", "không chỉ", "hơn cả")
_VI_INTENSIFIERS_HIGH = {
    "quá", "rất", "cực", "cực_kỳ", "vãi", "vcl", "vl",
    "lắm", "thiệt_sự", "thực_sự", "dữ_luôn", "vô_cùng", "luôn", "thêm",
}
_VI_INTENSIFIERS_LOW = {"hơi", "tạm", "khá", "chút", "tí"}

# --- Từ điển sắc thái Tiếng Anh (MXH / TikTok) ---
_EN_POSITIVE_LEXICON: dict[str, float] = {
    "peak": 1.7, "peakk": 1.8, "peakkk": 1.8, "peakkkk": 1.8, "peaak": 1.8,
    "fire": 1.6, "smooth": 1.5, "cute": 1.5, "good": 1.4, "great": 1.6,
    "amazing": 1.7, "incredible": 1.8, "awesome": 1.7, "love": 1.6, "loved": 1.4,
    "goated": 1.8, "goat": 1.5, "masterpiece": 1.8, "best": 1.6, "happy": 1.5,
    "merry": 1.2, "perfect": 1.7, "cool": 1.3, "nice": 1.3, "wonderful": 1.6,
    "brilliant": 1.6, "fantastic": 1.7, "clean": 1.2, "insane": 1.3, "crazy": 1.0,
    "wow": 1.2, "gold": 1.2, "enjoy": 1.3, "excited": 1.4, "favorite": 1.4,
    "w": 1.1, "🔥": 1.4, "❤️": 1.3, "💕": 1.2, "😍": 1.4, "🥰": 1.3,
    "👏": 1.1, "👍": 1.0, "🤩": 1.4, "✨": 0.9, "🎬": 0.6, "🍿": 0.6,
}

_EN_NEGATIVE_LEXICON: dict[str, float] = {
    "scared": 1.6, "afraid": 1.5, "fear": 1.5, "terrified": 1.7, "creepy": 1.4,
    "hate": 1.8, "hated": 1.6, "trash": 1.8, "garbage": 1.8, "mid": 1.4,
    "bad": 1.5, "awful": 1.7, "boring": 1.5, "sad": 1.5, "terrible": 1.8,
    "worst": 1.8, "cringe": 1.6, "ugly": 1.5, "cry": 1.2, "crying": 1.1,
    "disappointed": 1.6, "disappointing": 1.6, "angry": 1.7, "mad": 1.5,
    "annoying": 1.5, "pathetic": 1.7, "ruined": 1.6, "fail": 1.5, "sucks": 1.7,
    "stupid": 1.5, "dumb": 1.4, "horrible": 1.8, "disgusting": 1.9,
    "🤢": 1.7, "🤮": 1.8, "😡": 1.7, "😠": 1.6, "🤬": 1.8, "👎": 1.4, "😢": 1.1,
}

_EN_NEGATION_WORDS = {
    "not", "no", "never", "don't", "dont", "doesn't", "doesnt",
    "didn't", "didnt", "isn't", "isnt", "aren't", "arent",
    "can't", "cant", "won't", "wont", "nothing", "none",
}
_EN_INTENSIFIERS_HIGH = {
    "so", "very", "really", "super", "freaking", "fucking",
    "fuckin", "literally", "damn", "holy", "extremely", "truly",
}
_EN_INTENSIFIERS_LOW = {"slightly", "kinda", "kind", "bit", "little"}


class BilingualLexiconScorer:
    """Mô hình Rule-based / Lexicon song ngữ Việt – Anh xử lý phủ định & cường độ."""

    def score_one(
        self,
        segmented_text: str,
        cleaned_text: str = "",
        language: str = "vi",
    ) -> dict[str, Any]:
        if language == "en":
            return self._score_en(cleaned_text or segmented_text)
        return self._score_vi(segmented_text, cleaned_text)

    def _score_vi(self, segmented_text: str, cleaned_text: str) -> dict[str, Any]:
        raw_src = segmented_text.strip() or cleaned_text.strip()
        lower_full = (cleaned_text or segmented_text).lower()
        tokens = [t.lower() for t in raw_src.split() if t.strip()]

        pos_sum = 0.0
        neg_sum = 0.0
        has_escalation_neg = any(m in lower_full for m in _VI_ESCALATION_MARKERS)

        for idx, tok in enumerate(tokens):
            clean_tok = tok.strip(".,!?;:\"'()[]{}")
            if not clean_tok:
                continue

            pos_w = _VI_POSITIVE_LEXICON.get(clean_tok, 0.0)
            neg_w = _VI_NEGATIVE_LEXICON.get(clean_tok, 0.0)
            if pos_w == 0.0 and neg_w == 0.0:
                continue

            multiplier = 1.0
            neighbors = tokens[max(0, idx - 2) : idx] + tokens[idx + 1 : min(len(tokens), idx + 3)]
            for nb in neighbors:
                nb_clean = nb.strip(".,!?;:\"'()[]{}")
                if nb_clean in _VI_INTENSIFIERS_HIGH:
                    multiplier = max(multiplier, 1.35)
                elif nb_clean in _VI_INTENSIFIERS_LOW:
                    multiplier = min(multiplier, 0.7)

            prev_tokens = [t.strip(".,!?;:\"'()[]{}") for t in tokens[max(0, idx - 2) : idx]]
            is_negated = any(pt in _VI_NEGATION_WORDS for pt in prev_tokens)
            if is_negated and neg_w > 0 and has_escalation_neg:
                is_negated = False

            if pos_w > 0:
                val = pos_w * multiplier
                if is_negated:
                    neg_sum += val * 0.9
                else:
                    pos_sum += val
            elif neg_w > 0:
                val = neg_w * multiplier
                if is_negated:
                    pos_sum += val * 0.4
                else:
                    neg_sum += val

        return self._to_probs(pos_sum, neg_sum)

    def _score_en(self, text: str) -> dict[str, Any]:
        raw_tokens = re.findall(r"[A-Za-z0-9']+|[^\w\s]", text.lower())
        tokens = [t for t in raw_tokens if t.strip()]
        lower_full = text.lower()

        pos_sum = 0.0
        neg_sum = 0.0
        has_no_more = "no more" in lower_full

        for idx, tok in enumerate(tokens):
            # Chuẩn hóa kéo dài ký tự kiểu "peakkkk" -> thử cả dạng gốc và dạng rút gọn
            collapsed = re.sub(r"(.)\1{2,}", r"\1\1", tok)
            collapsed_one = re.sub(r"(.)\1+", r"\1", tok)
            pos_w = (
                _EN_POSITIVE_LEXICON.get(tok, 0.0)
                or _EN_POSITIVE_LEXICON.get(collapsed, 0.0)
                or _EN_POSITIVE_LEXICON.get(collapsed_one, 0.0)
            )
            neg_w = (
                _EN_NEGATIVE_LEXICON.get(tok, 0.0)
                or _EN_NEGATIVE_LEXICON.get(collapsed, 0.0)
            )
            if pos_w == 0.0 and neg_w == 0.0:
                continue

            multiplier = 1.0
            neighbors = tokens[max(0, idx - 2) : idx] + tokens[idx + 1 : min(len(tokens), idx + 3)]
            for nb in neighbors:
                if nb in _EN_INTENSIFIERS_HIGH:
                    multiplier = max(multiplier, 1.35)
                elif nb in _EN_INTENSIFIERS_LOW:
                    multiplier = min(multiplier, 0.7)

            prev_tokens = tokens[max(0, idx - 2) : idx]
            is_negated = any(pt in _EN_NEGATION_WORDS for pt in prev_tokens)
            if tok == "love" and has_no_more:
                is_negated = True

            if pos_w > 0:
                val = pos_w * multiplier
                if is_negated:
                    neg_sum += val * 0.85
                else:
                    pos_sum += val
            elif neg_w > 0:
                val = neg_w * multiplier
                if is_negated:
                    pos_sum += val * 0.45
                else:
                    neg_sum += val

        return self._to_probs(pos_sum, neg_sum)

    @staticmethod
    def _to_probs(pos_sum: float, neg_sum: float) -> dict[str, Any]:
        net = pos_sum - neg_sum
        strength = pos_sum + neg_sum
        if strength < 0.15:
            probs = {"POS": 0.15, "NEG": 0.15, "NEU": 0.70}
        else:
            raw_pos = math.exp(min(6.0, pos_sum * 1.15))
            raw_neg = math.exp(min(6.0, neg_sum * 1.15))
            raw_neu = math.exp(max(0.0, 1.2 - 0.35 * abs(net)))
            total = raw_pos + raw_neg + raw_neu
            probs = {
                "POS": round(raw_pos / total, 4),
                "NEG": round(raw_neg / total, 4),
                "NEU": round(raw_neu / total, 4),
            }
        best_label = max(probs, key=probs.get)  # type: ignore[arg-type]
        return {
            "label": best_label,
            "scores": probs,
            "pos_lex_strength": round(pos_sum, 4),
            "neg_lex_strength": round(neg_sum, 4),
        }

    def predict_batch(
        self,
        segmented_texts: Sequence[str],
        cleaned_texts: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        cleaned_list = (
            list(cleaned_texts) if cleaned_texts is not None else [""] * len(segmented_texts)
        )
        lang_list = list(languages) if languages is not None else ["vi"] * len(segmented_texts)
        return [
            self.score_one(seg, cln, language=lang)
            for seg, cln, lang in zip(segmented_texts, cleaned_list, lang_list)
        ]


_FALLBACK_VI_POLARITY_DATA: list[tuple[str, str]] = [
    ("sản_phẩm tuyệt_vời rất thích ủng_hộ", "POS"),
    ("hay quá giỏi thật luôn xuất_sắc", "POS"),
    ("dễ_thương tử_tế ấm_áp quá đi", "POS"),
    ("đẹp_trai duyên_dáng nói_chuyện có_duyên", "POS"),
    ("rất hài_lòng chất_lượng tốt đáng tiền", "POS"),
    ("mừng cho bạn mong sự_nghiệp thăng_hoa", "POS"),
    ("quá đỉnh luôn cảm_ơn nhiều", "POS"),
    ("bình_thường thôi không có gì đặc_biệt", "NEU"),
    ("hình_như hôm_qua mới đăng bài này", "NEU"),
    ("cho hỏi cái này ở đâu vậy ạ", "NEU"),
    ("vẫn chưa xem tập mới nhất", "NEU"),
    ("đây là chương_trình gì thế mọi người", "NEU"),
    ("ghét quá coi riết thù luôn", "NEG"),
    ("vô_duyên thực_sự mắc_ói quá đi", "NEG"),
    ("càng xem càng thấy ghét khó_chịu", "NEG"),
    ("chán quá tụt năng_lượng ghê", "NEG"),
    ("giả_tạo lố_bịch thất_vọng toàn_tập", "NEG"),
    ("tệ hại kém chất_lượng không nên mua", "NEG"),
    ("đơm_đặt tự_ti ganh_tị trù_ẻo người_ta", "NEG"),
]

_FALLBACK_VI_EMOTION_DATA: list[tuple[str, str]] = [
    ("tức_giận cay_cú ghét thù thật_sự", "Anger"),
    ("bực_mình quá sao lại làm vậy", "Anger"),
    ("buồn quá khóc mất thôi thương thật", "Sadness"),
    ("tội_nghiệp cô_đơn thất_vọng buồn_bã", "Sadness"),
    ("sợ quá lo_lắng hồi_hộp run luôn", "Fear"),
    ("hoang_mang lo_sợ không dám xem", "Fear"),
    ("vui quá tuyệt_vời hạnh_phúc hài_lòng", "Enjoyment"),
    ("thích quá cười xỉu luôn hay thật", "Enjoyment"),
    ("mắc_ói vô_duyên kinh_tởm ghê_tởm chán_ghét", "Disgust"),
    ("lố_bịch giả_tạo khinh thường", "Disgust"),
    ("ủa thật hả trời bất_ngờ ngạc_nhiên quá", "Surprise"),
    ("không ngờ luôn đỉnh vậy sao", "Surprise"),
    ("bình_thường hôm_nay thứ mấy vậy", "Other"),
    ("ai biết tên bài hát này không", "Other"),
]

_FALLBACK_EN_POLARITY_DATA: list[tuple[str, str]] = [
    ("this is peak edit so freaking good", "POS"),
    ("holy shit this is incredible masterpiece", "POS"),
    ("so cute fire smooth transitions love it", "POS"),
    ("awesome video amazing work best ever", "POS"),
    ("merry christmas happy excited great edit", "POS"),
    ("different mask same task just watching", "NEU"),
    ("what song is this in the background", "NEU"),
    ("see you in december when the movie drops", "NEU"),
    ("how does this say only one repost", "NEU"),
    ("i am so fucking scared terrified", "NEG"),
    ("this is trash boring mid awful edit", "NEG"),
    ("hate this so disappointed worst video", "NEG"),
    ("cringe disgusting pathetic terrible waste", "NEG"),
    ("no more love so sad crying", "NEG"),
]

_SUPPLEMENT_EN_EMOTION_DATA: list[tuple[str, str]] = [
    ("this makes me so angry mad furious hate it", "Anger"),
    ("wtf stop doing this so annoying pissed off", "Anger"),
    ("shut up stop ruining everything i am furious", "Anger"),
    ("why would anyone defend this garbage makes my blood boil", "Anger"),
    ("i am so sad crying miss him heartbroken", "Sadness"),
    ("so disappointed lonely depressing makes me cry", "Sadness"),
    ("i feel empty today nothing hurts more than this", "Sadness"),
    ("rest in peace crying watching this scene again", "Sadness"),
    ("i am so fucking scared terrified creepy afraid", "Fear"),
    ("this gives me anxiety so nervous worried fear", "Fear"),
    ("chills down my spine horrifying nightmare fuel", "Fear"),
    ("genuinely terrified of what happens next so eerie", "Fear"),
    ("so cute peak edit fire smooth love this amazing", "Enjoyment"),
    ("holy shit this is incredible so freaking good happy", "Enjoyment"),
    ("best edit ever made my whole day smiling nonstop", "Enjoyment"),
    ("goosebumps pure perfection i love this so much", "Enjoyment"),
    ("disgusting trash cringe pathetic gross awful", "Disgust"),
    ("ew hate this fake garbage makes me sick", "Disgust"),
    ("repulsive behavior nauseating worst take ever", "Disgust"),
    ("so cringe and toxic i want to throw up", "Disgust"),
    ("vile nasty attitude completely grossed out", "Disgust"),
    ("literally unwatchable garbage content gross", "Disgust"),
    ("no way what is this i am witnessing omg shocked", "Surprise"),
    ("unbelievable plot twist unexpected surprise", "Surprise"),
    ("wait what just happened my jaw dropped", "Surprise"),
    ("did not see that coming at all insane twist", "Surprise"),
    ("different mask same task see you in december", "Other"),
    ("what is the name of this song or movie", "Other"),
    ("does anyone know which episode this clip is from", "Other"),
    ("watching this at 2am on my feed today", "Other"),
    ("here is the timestamp for the background audio", "Other"),
    ("commenting to save this video for later", "Other"),
]

_EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
_REPEAT_CHAR_PATTERN = re.compile(r"(.)\1{2,}")


class _SocialLexiconFeatureTransformer(BaseEstimator, TransformerMixin):
    """Trích xuất 15 đặc trưng phi âm [0, 1] từ từ điển sắc thái, phủ định, cường độ và tín hiệu MXH."""

    FEATURE_NAMES: tuple[str, ...] = (
        "pos_lex_score",
        "neg_lex_score",
        "neu_lex_score",
        "pos_lex_strength",
        "neg_lex_strength",
        "lex_margin_pos",
        "lex_margin_neg",
        "has_negation",
        "has_intensifier",
        "excl_ratio",
        "quest_ratio",
        "upper_ratio",
        "char_len_norm",
        "word_count_norm",
        "emoji_repeat_signal",
    )

    def __init__(self, language: str = "vi") -> None:
        self.language = "en" if language == "en" else "vi"

    def fit(self, X: Sequence[str], y: Any = None) -> "_SocialLexiconFeatureTransformer":
        self.feature_names_ = np.asarray(self.FEATURE_NAMES, dtype=object)
        return self

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.FEATURE_NAMES, dtype=object)

    def transform(self, X: Sequence[str]) -> sparse.csr_matrix:
        scorer = BilingualLexiconScorer()
        neg_vocab = _EN_NEGATION_WORDS if self.language == "en" else _VI_NEGATION_WORDS
        int_vocab = _EN_INTENSIFIERS_HIGH if self.language == "en" else _VI_INTENSIFIERS_HIGH

        n_samples = len(X)
        mat = np.zeros((n_samples, len(self.FEATURE_NAMES)), dtype=np.float64)

        for row_idx, raw_text in enumerate(X):
            txt = str(raw_text or "")
            low = txt.lower()
            lex = scorer.score_one(segmented_text=txt, cleaned_text=txt, language=self.language)
            scores = lex["scores"]
            p_pos = float(scores.get("POS", 0.0))
            p_neg = float(scores.get("NEG", 0.0))
            p_neu = float(scores.get("NEU", 0.0))
            pos_str = min(1.0, float(lex.get("pos_lex_strength", 0.0)) / 6.0)
            neg_str = min(1.0, float(lex.get("neg_lex_strength", 0.0)) / 6.0)

            tokens = [t.strip(".,!?;:\"'()[]{}") for t in low.split() if t.strip()]
            has_neg = 1.0 if any(tok in neg_vocab for tok in tokens) else 0.0
            has_int = 1.0 if any(tok in int_vocab for tok in tokens) else 0.0

            excl_ratio = min(1.0, txt.count("!") / 3.0)
            quest_ratio = min(1.0, txt.count("?") / 3.0)
            alpha_chars = [ch for ch in txt if ch.isalpha()]
            upper_ratio = (
                sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)
                if alpha_chars
                else 0.0
            )
            char_len_norm = min(1.0, len(txt) / 160.0)
            word_count_norm = min(1.0, len(tokens) / 30.0)
            has_emoji_or_repeat = (
                1.0
                if (_EMOJI_PATTERN.search(txt) or _REPEAT_CHAR_PATTERN.search(low))
                else 0.0
            )

            mat[row_idx] = np.clip(
                [
                    p_pos,
                    p_neg,
                    p_neu,
                    pos_str,
                    neg_str,
                    max(0.0, p_pos - p_neg),
                    max(0.0, p_neg - p_pos),
                    has_neg,
                    has_int,
                    excl_ratio,
                    quest_ratio,
                    upper_ratio,
                    char_len_norm,
                    word_count_norm,
                    has_emoji_or_repeat,
                ],
                0.0,
                1.0,
            )

        return sparse.csr_matrix(mat, dtype=np.float64)


class _SVDSocialFeatureExtractor(BaseEstimator, TransformerMixin):
    """Chỉ nén phần TF-IDF thưa bằng TruncatedSVD và giữ nguyên vẹn 15 đặc trưng từ điển/MXH."""

    def __init__(
        self,
        n_components: int = 96,
        n_social_features: int = 15,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.n_social_features = n_social_features
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> "_SVDSocialFeatureExtractor":
        n_samples, total_cols = X.shape
        n_soc = min(self.n_social_features, max(0, total_cols - 2))
        self.n_social_used_ = n_soc
        text_cols = total_cols - n_soc
        if text_cols >= 3 and n_samples >= 3:
            n_comp = min(self.n_components, text_cols - 1, n_samples - 1)
            self.svd_: TruncatedSVD | None = TruncatedSVD(
                n_components=n_comp, random_state=self.random_state
            )
            self.svd_.fit(X[:, :text_cols])
        else:
            self.svd_ = None
        return self

    def transform(self, X: Any) -> np.ndarray:
        total_cols = X.shape[1]
        n_soc = int(getattr(self, "n_social_used_", 0))
        text_cols = total_cols - n_soc
        if self.svd_ is not None:
            left = self.svd_.transform(X[:, :text_cols])
        else:
            raw_left = X[:, :text_cols]
            left = raw_left.toarray() if sparse.issparse(raw_left) else np.asarray(raw_left)
        if n_soc > 0:
            raw_right = X[:, text_cols:]
            right = raw_right.toarray() if sparse.issparse(raw_right) else np.asarray(raw_right)
            return np.hstack([left, right])
        return left


class _NBSVMClassifier(BaseEstimator, ClassifierMixin):
    """NBSVM đa lớp OvR (Wang & Manning 2012): nhân log-count ratio Naive Bayes trước LinearSVC."""

    def __init__(
        self,
        C: float = 0.5,
        alpha: float = 1.0,
        class_weight: str | None = None,
        cv: int = 2,
        random_state: int = 42,
    ) -> None:
        self.C = C
        self.alpha = alpha
        self.class_weight = class_weight
        self.cv = cv
        self.random_state = random_state

    def fit(self, X: Any, y: Sequence[str]) -> "_NBSVMClassifier":
        y_arr = np.asarray(list(y), dtype=object)
        self.classes_ = np.unique(y_arr)
        self.models_: dict[str, tuple[np.ndarray, Any]] = {}

        X_csr = X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X)
        for cls in self.classes_:
            mask = y_arr == cls
            y_bin = mask.astype(np.int32)
            pos_cnt = int(mask.sum())
            neg_cnt = int((~mask).sum())

            p = np.asarray(X_csr[mask].sum(axis=0)).ravel() + float(self.alpha)
            q = np.asarray(X_csr[~mask].sum(axis=0)).ravel() + float(self.alpha)
            p_norm = p / max(float(p.sum()), 1e-12)
            q_norm = q / max(float(q.sum()), 1e-12)
            r = np.log(np.clip(p_norm, 1e-12, None) / np.clip(q_norm, 1e-12, None))

            X_nb = X_csr.multiply(r).tocsr()
            eff_cv = min(int(self.cv), pos_cnt, neg_cnt)
            if eff_cv >= 2:
                base_svc = LinearSVC(
                    C=self.C,
                    class_weight=self.class_weight,
                    max_iter=3000,
                    random_state=self.random_state,
                )
                clf: Any = CalibratedClassifierCV(base_svc, cv=eff_cv, method="sigmoid")
            else:
                clf = LogisticRegression(
                    C=self.C,
                    max_iter=1000,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                )
            clf.fit(X_nb, y_bin)
            self.models_[str(cls)] = (r, clf)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X_csr = X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X)
        n_samples = X_csr.shape[0]
        n_classes = len(self.classes_)
        if n_samples == 0:
            return np.zeros((0, n_classes), dtype=np.float64)

        probs = np.zeros((n_samples, n_classes), dtype=np.float64)
        for col_idx, cls in enumerate(self.classes_):
            r, clf = self.models_[str(cls)]
            X_nb = X_csr.multiply(r).tocsr()
            p_mat = clf.predict_proba(X_nb)
            clf_classes = list(clf.classes_)
            if 1 in clf_classes:
                probs[:, col_idx] = p_mat[:, clf_classes.index(1)]
            elif p_mat.shape[1] >= 2:
                probs[:, col_idx] = p_mat[:, -1]
            else:
                probs[:, col_idx] = p_mat[:, 0]

        row_sums = probs.sum(axis=1, keepdims=True)
        safe = row_sums > 0
        out = np.full_like(probs, 1.0 / max(1, n_classes))
        np.divide(probs, row_sums, out=out, where=safe)
        return out

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class _SubsampledClassifier(BaseEstimator, ClassifierMixin):
    """Lấy mẫu phân tầng tối đa max_samples dòng trước khi fit mô hình phi tuyến nặng trên CPU."""

    def __init__(
        self,
        estimator: Any,
        max_samples: int = 3500,
        random_state: int = 42,
    ) -> None:
        self.estimator = estimator
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X: Any, y: Sequence[str]) -> "_SubsampledClassifier":
        y_arr = np.asarray(list(y), dtype=object)
        n_total = len(y_arr)
        self.estimator_ = clone(self.estimator)
        if n_total > self.max_samples:
            try:
                sub_idx, _ = train_test_split(
                    np.arange(n_total),
                    train_size=self.max_samples,
                    random_state=self.random_state,
                    stratify=y_arr,
                )
            except Exception:
                rng = np.random.default_rng(self.random_state)
                sub_idx = rng.choice(n_total, size=self.max_samples, replace=False)
            self.estimator_.fit(X[sub_idx], y_arr[sub_idx])
        else:
            self.estimator_.fit(X, y_arr)
        self.classes_ = np.asarray(self.estimator_.classes_, dtype=object)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        return self.estimator_.predict(X)


class _OOFStackingEnsemble(BaseEstimator):
    """Meta-learner LogisticRegression trên xác suất dự đoán của các mô hình ML thành phần."""

    def __init__(
        self,
        base_model_names: Sequence[str],
        classes: Sequence[str],
        base_models: dict[str, Any] | None = None,
        meta_model: LogisticRegression | None = None,
    ) -> None:
        self.base_model_names = list(base_model_names)
        self.classes_ = np.asarray(list(classes), dtype=object)
        self.base_models = dict(base_models or {})
        self.meta_model = meta_model

    @staticmethod
    def _probs_dict_to_meta_features(
        probs_by_model: dict[str, np.ndarray],
        base_model_names: Sequence[str],
    ) -> np.ndarray:
        blocks = [probs_by_model[name] for name in base_model_names]
        return np.hstack(blocks)

    @staticmethod
    def _extract_aligned_probs(clf: Any, X: Any, target_classes: Sequence[str]) -> np.ndarray:
        raw_probs = clf.predict_proba(X)
        clf_classes = [str(c) for c in clf.classes_]
        cls_to_col = {cls_name: idx for idx, cls_name in enumerate(clf_classes)}
        aligned = np.zeros((raw_probs.shape[0], len(target_classes)), dtype=np.float64)
        for tgt_idx, cls_name in enumerate(target_classes):
            src_idx = cls_to_col.get(str(cls_name))
            if src_idx is not None:
                aligned[:, tgt_idx] = raw_probs[:, src_idx]
        return aligned

    def predict_proba_from_base_probs(
        self,
        base_prob_dicts: dict[str, list[dict[str, Any]]],
    ) -> np.ndarray:
        assert self.meta_model is not None
        classes = [str(c) for c in self.classes_]
        first_list = base_prob_dicts[self.base_model_names[0]]
        n = len(first_list)
        if n == 0:
            return np.zeros((0, len(classes)), dtype=np.float64)
        blocks: list[np.ndarray] = []
        for name in self.base_model_names:
            rows = base_prob_dicts[name]
            mat = np.array(
                [[float(item["scores"].get(c, 0.0)) for c in classes] for item in rows],
                dtype=np.float64,
            )
            blocks.append(mat)
        meta_X = np.hstack(blocks)
        return self._extract_aligned_probs(self.meta_model, meta_X, classes)

    def predict_proba(self, X: Any) -> np.ndarray:
        assert self.meta_model is not None
        classes = [str(c) for c in self.classes_]
        probs_by_model = {
            name: self._extract_aligned_probs(self.base_models[name], X, classes)
            for name in self.base_model_names
        }
        meta_X = self._probs_dict_to_meta_features(probs_by_model, self.base_model_names)
        return self._extract_aligned_probs(self.meta_model, meta_X, classes)

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)
        best_idx = np.argmax(probs, axis=1)
        return self.classes_[best_idx]


class _WeightedSoftVoteEnsemble(BaseEstimator):
    """Ensemble bỏ phiếu mềm có trọng số theo OOF Macro-F1."""

    def __init__(
        self,
        base_model_names: Sequence[str],
        weights: dict[str, float],
        classes: Sequence[str],
        base_models: dict[str, Any] | None = None,
    ) -> None:
        self.base_model_names = list(base_model_names)
        self.weights = dict(weights)
        self.classes_ = np.asarray(list(classes), dtype=object)
        self.base_models = dict(base_models or {})

    def predict_proba_from_base_probs(
        self,
        base_prob_dicts: dict[str, list[dict[str, Any]]],
    ) -> np.ndarray:
        classes = [str(c) for c in self.classes_]
        n = len(base_prob_dicts[self.base_model_names[0]])
        out = np.zeros((n, len(classes)), dtype=np.float64)
        total_w = 0.0
        for name in self.base_model_names:
            w = max(0.05, float(self.weights.get(name, 1.0)))
            rows = base_prob_dicts[name]
            mat = np.array(
                [[float(item["scores"].get(c, 0.0)) for c in classes] for item in rows],
                dtype=np.float64,
            )
            out += w * mat
            total_w += w
        if total_w > 0:
            out /= total_w
        return out

    def predict_proba(self, X: Any) -> np.ndarray:
        classes = [str(c) for c in self.classes_]
        n = X.shape[0]
        out = np.zeros((n, len(classes)), dtype=np.float64)
        total_w = 0.0
        for name in self.base_model_names:
            w = max(0.05, float(self.weights.get(name, 1.0)))
            probs = _OOFStackingEnsemble._extract_aligned_probs(self.base_models[name], X, classes)
            out += w * probs
            total_w += w
        if total_w > 0:
            out /= total_w
        return out

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


for _ml_cls in (
    _SocialLexiconFeatureTransformer,
    _SVDSocialFeatureExtractor,
    _NBSVMClassifier,
    _SubsampledClassifier,
    _OOFStackingEnsemble,
    _WeightedSoftVoteEnsemble,
):
    _ml_cls.__module__ = "src.models.classical_ml"


def _fast_segment_batch(texts: Sequence[str]) -> list[str]:
    try:
        from underthesea import word_tokenize

        return [word_tokenize(str(t), format="text") for t in texts]
    except Exception:
        return [str(t) for t in texts]


def _normalize_corpus_texts(
    texts: Sequence[str],
    labels: Sequence[str],
    language: str,
    segment_vi: bool = True,
) -> tuple[list[str], list[str]]:
    """Chuẩn hóa đồng bộ clean_comment_text() giữa lúc train và inference."""
    clean_pairs: list[tuple[str, str]] = []
    for txt, lbl in zip(texts, labels):
        cleaned = clean_comment_text(str(txt), normalize_teencode=True, language=language)
        if cleaned:
            clean_pairs.append((cleaned, str(lbl)))
    if not clean_pairs:
        return [], []
    norm_texts = [t for t, _ in clean_pairs]
    norm_labels = [l for _, l in clean_pairs]
    if language == "vi" and segment_vi:
        norm_texts = _fast_segment_batch(norm_texts)
    return norm_texts, norm_labels


def _build_feature_union(language: str, sample_count: int = 1000) -> FeatureUnion:
    min_word_df = 2 if sample_count >= 200 else 1
    min_char_df = 3 if sample_count >= 200 else 1
    return FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    token_pattern=r"(?u)\b[\w_]{1,}\b",
                    ngram_range=(1, 3),
                    min_df=min_word_df,
                    max_df=0.95,
                    max_features=35000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=min_char_df,
                    max_df=0.95,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
            (
                "social_lex",
                _SocialLexiconFeatureTransformer(language=language),
            ),
        ]
    )


def _load_vi_corpora() -> tuple[
    list[str], list[str], list[str], list[str], list[str], list[str], list[str], list[str]
]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    pol_train_texts: list[str] = []
    pol_train_labels: list[str] = []
    pol_test_texts: list[str] = []
    pol_test_labels: list[str] = []
    emo_train_texts: list[str] = []
    emo_train_labels: list[str] = []
    emo_test_texts: list[str] = []
    emo_test_labels: list[str] = []

    def _norm_pol(val: Any) -> str | None:
        up = str(val or "").strip().upper()
        if up in {"POS", "POSITIVE", "2"}:
            return "POS"
        if up in {"NEG", "NEGATIVE", "0"}:
            return "NEG"
        if up in {"NEU", "NEUTRAL", "1"}:
            return "NEU"
        return None

    try:
        train_csv = hf_hub_download(
            repo_id="h-i-e-u/vietnamese-SA-dataset",
            filename="train_phobert.csv",
            repo_type="dataset",
        )
        test_csv = hf_hub_download(
            repo_id="h-i-e-u/vietnamese-SA-dataset",
            filename="test_phobert.csv",
            repo_type="dataset",
        )
        df_tr = pd.read_csv(train_csv)
        for txt, lbl in zip(df_tr["text"].fillna(""), df_tr["label"].fillna("")):
            norm_lbl = _norm_pol(lbl)
            clean_txt = str(txt).replace("_", " ").strip()
            if clean_txt and norm_lbl:
                pol_train_texts.append(clean_txt)
                pol_train_labels.append(norm_lbl)

        df_te = pd.read_csv(test_csv)
        for txt, lbl in zip(df_te["text"].fillna(""), df_te["label"].fillna("")):
            norm_lbl = _norm_pol(lbl)
            clean_txt = str(txt).replace("_", " ").strip()
            if clean_txt and norm_lbl:
                pol_test_texts.append(clean_txt)
                pol_test_labels.append(norm_lbl)
    except Exception as exc:
        print(f"[Cảnh báo] Không tải được h-i-e-u/vietnamese-SA-dataset: {exc}")

    vsmec_pol_map = {
        "Enjoyment": "POS",
        "Anger": "NEG",
        "Disgust": "NEG",
        "Sadness": "NEG",
        "Fear": "NEG",
        "Surprise": "NEU",
        "Other": "NEU",
    }
    try:
        for split_name, fname in (
            ("train", "train.csv"),
            ("validation", "validation.csv"),
            ("test", "test.csv"),
        ):
            csv_path = hf_hub_download(
                repo_id="ura-hcmut/UIT-VSMEC",
                filename=fname,
                repo_type="dataset",
            )
            df_v = pd.read_csv(csv_path)
            for emo_raw, sent_raw in zip(
                df_v["Emotion"].fillna("Other"), df_v["Sentence"].fillna("")
            ):
                txt = str(sent_raw).strip()
                emo_name = str(emo_raw).strip()
                if emo_name not in EMOTION_LABELS:
                    emo_name = "Other"
                if not txt:
                    continue
                mapped_pol = vsmec_pol_map.get(emo_name, "NEU")
                if split_name == "test":
                    emo_test_texts.append(txt)
                    emo_test_labels.append(emo_name)
                    pol_test_texts.append(txt)
                    pol_test_labels.append(mapped_pol)
                else:
                    emo_train_texts.append(txt)
                    emo_train_labels.append(emo_name)
                    pol_train_texts.append(txt)
                    pol_train_labels.append(mapped_pol)
    except Exception as exc:
        print(f"[Cảnh báo] Không tải được ura-hcmut/UIT-VSMEC: {exc}")

    return (
        pol_train_texts,
        pol_train_labels,
        pol_test_texts,
        pol_test_labels,
        emo_train_texts,
        emo_train_labels,
        emo_test_texts,
        emo_test_labels,
    )


def _fetch_text_lines(url: str, timeout: int = 20) -> list[str]:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return [line.strip() for line in resp.text.splitlines()]


def _load_dair_ai_emotion(
    max_train: int = 6000,
    max_test: int = 1500,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Tải bộ dair-ai/emotion (6 cảm xúc chuẩn) với fallback an toàn không cần pyarrow."""
    label_id_map = {
        0: "Sadness",
        1: "Enjoyment",
        2: "Enjoyment",
        3: "Anger",
        4: "Fear",
        5: "Surprise",
        "sadness": "Sadness",
        "joy": "Enjoyment",
        "love": "Enjoyment",
        "anger": "Anger",
        "fear": "Fear",
        "surprise": "Surprise",
    }

    def _map_lbl(raw: Any) -> str | None:
        if isinstance(raw, (int, np.integer)):
            return label_id_map.get(int(raw))
        low = str(raw).strip().lower()
        if low.isdigit():
            return label_id_map.get(int(low))
        return label_id_map.get(low)

    tr_x: list[str] = []
    tr_y: list[str] = []
    te_x: list[str] = []
    te_y: list[str] = []

    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download

        tr_pq = hf_hub_download(
            repo_id="dair-ai/emotion",
            filename="split/train-00000-of-00001.parquet",
            repo_type="dataset",
        )
        te_pq = hf_hub_download(
            repo_id="dair-ai/emotion",
            filename="split/test-00000-of-00001.parquet",
            repo_type="dataset",
        )
        df_tr = pd.read_parquet(tr_pq)
        df_te = pd.read_parquet(te_pq)
        for txt, lbl in zip(df_tr["text"].fillna(""), df_tr["label"]):
            m = _map_lbl(lbl)
            if str(txt).strip() and m:
                tr_x.append(str(txt).strip())
                tr_y.append(m)
                if len(tr_x) >= max_train:
                    break
        for txt, lbl in zip(df_te["text"].fillna(""), df_te["label"]):
            m = _map_lbl(lbl)
            if str(txt).strip() and m:
                te_x.append(str(txt).strip())
                te_y.append(m)
                if len(te_x) >= max_test:
                    break
        if tr_x and te_x:
            return tr_x, tr_y, te_x, te_y
    except Exception:
        tr_x.clear()
        tr_y.clear()
        te_x.clear()
        te_y.clear()

    try:
        base_api = "https://datasets-server.huggingface.co/rows"
        for split_name, limit_n, out_x, out_y in (
            ("train", min(max_train, 3000), tr_x, tr_y),
            ("test", min(max_test, 1000), te_x, te_y),
        ):
            offset = 0
            while offset < limit_n:
                batch_len = min(100, limit_n - offset)
                resp = requests.get(
                    base_api,
                    params={
                        "dataset": "dair-ai/emotion",
                        "config": "split",
                        "split": split_name,
                        "offset": offset,
                        "length": batch_len,
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                rows = resp.json().get("rows") or []
                if not rows:
                    break
                for item in rows:
                    row = item.get("row") or {}
                    txt = str(row.get("text") or "").strip()
                    m = _map_lbl(row.get("label"))
                    if txt and m:
                        out_x.append(txt)
                        out_y.append(m)
                offset += len(rows)
                if len(rows) < batch_len:
                    break
        if tr_x and te_x:
            return tr_x, tr_y, te_x, te_y
    except Exception as exc:
        print(f"[Cảnh báo] Không tải được dair-ai/emotion: {exc}")

    return [], [], [], []


def _load_en_corpora() -> tuple[
    list[str], list[str], list[str], list[str], list[str], list[str], list[str], list[str]
]:
    """Tải dữ liệu Sentiment & Emotion EN (TweetEval + dair-ai/emotion với fallback an toàn)."""
    base = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets"
    pol_map = {"0": "NEG", "1": "NEU", "2": "POS"}
    emo_map = {"0": "Anger", "1": "Enjoyment", "2": "Enjoyment", "3": "Sadness"}

    pol_train_x: list[str] = []
    pol_train_y: list[str] = []
    pol_test_x: list[str] = []
    pol_test_y: list[str] = []
    emo_train_x: list[str] = []
    emo_train_y: list[str] = []
    emo_test_x: list[str] = []
    emo_test_y: list[str] = []

    try:
        tr_t = _fetch_text_lines(f"{base}/sentiment/train_text.txt")
        tr_l = _fetch_text_lines(f"{base}/sentiment/train_labels.txt")
        te_t = _fetch_text_lines(f"{base}/sentiment/test_text.txt")
        te_l = _fetch_text_lines(f"{base}/sentiment/test_labels.txt")
        for txt, lbl in zip(tr_t, tr_l):
            if txt and lbl in pol_map:
                pol_train_x.append(txt)
                pol_train_y.append(pol_map[lbl])
        for txt, lbl in zip(te_t, te_l):
            if txt and lbl in pol_map:
                pol_test_x.append(txt)
                pol_test_y.append(pol_map[lbl])
    except Exception as exc:
        print(f"[Cảnh báo] Không tải được TweetEval sentiment EN: {exc}")

    try:
        etr_t = _fetch_text_lines(f"{base}/emotion/train_text.txt")
        etr_l = _fetch_text_lines(f"{base}/emotion/train_labels.txt")
        ete_t = _fetch_text_lines(f"{base}/emotion/test_text.txt")
        ete_l = _fetch_text_lines(f"{base}/emotion/test_labels.txt")
        for txt, lbl in zip(etr_t, etr_l):
            if txt and lbl in emo_map:
                emo_train_x.append(txt)
                emo_train_y.append(emo_map[lbl])
        for txt, lbl in zip(ete_t, ete_l):
            if txt and lbl in emo_map:
                emo_test_x.append(txt)
                emo_test_y.append(emo_map[lbl])
    except Exception as exc:
        print(f"[Cảnh báo] Không tải được TweetEval emotion EN: {exc}")

    d_tr_x, d_tr_y, d_te_x, d_te_y = _load_dair_ai_emotion()
    if d_tr_x:
        emo_train_x.extend(d_tr_x)
        emo_train_y.extend(d_tr_y)
    if d_te_x:
        emo_test_x.extend(d_te_x)
        emo_test_y.extend(d_te_y)

    return (
        pol_train_x,
        pol_train_y,
        pol_test_x,
        pol_test_y,
        emo_train_x,
        emo_train_y,
        emo_test_x,
        emo_test_y,
    )


def _evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    """Tính Accuracy, Macro-F1, Weighted-F1, Per-Class P/R/F1 và Confusion Matrix."""
    present_labels = [lbl for lbl in labels if lbl in set(y_true) or lbl in set(y_pred)]
    if not present_labels:
        present_labels = list(labels)

    report = classification_report(
        y_true,
        y_pred,
        labels=present_labels,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    per_class: dict[str, dict[str, Any]] = {}
    for lbl in present_labels:
        row = report.get(lbl, {})
        per_class[lbl] = {
            "precision": round(float(row.get("precision", 0.0)), 4),
            "recall": round(float(row.get("recall", 0.0)), 4),
            "f1": round(float(row.get("f1-score", 0.0)), 4),
            "support": int(row.get("support", 0)),
        }

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(present_labels),
            "matrix": [[int(v) for v in row] for row in cm.tolist()],
        },
    }


def _extract_top_features(
    vectorizer: FeatureUnion,
    linear_clf: Any,
    top_k: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Rút trích các đặc trưng có trọng số lớn nhất cho từng lớp từ mô hình tuyến tính."""
    try:
        feat_names = vectorizer.get_feature_names_out()
        coef = getattr(linear_clf, "coef_", None)
        if coef is None:
            return {}
        classes = [str(c) for c in linear_clf.classes_]
        top_map: dict[str, list[dict[str, Any]]] = {}
        for idx, cls_name in enumerate(classes):
            row = coef[idx] if coef.ndim == 2 else coef
            top_indices = np.argsort(row)[::-1][:top_k]
            top_map[cls_name] = [
                {
                    "feature": str(feat_names[f_i]),
                    "weight": round(float(row[f_i]), 4),
                }
                for f_i in top_indices
                if float(row[f_i]) > 0
            ]
        return top_map
    except Exception:
        return {}


def _build_tree_pipeline() -> Pipeline:
    """Pipeline SVD (chỉ phần TF-IDF) + 15 Social Features nguyên bản + HistGradientBoosting."""
    return Pipeline(
        [
            ("svd_soc", _SVDSocialFeatureExtractor(n_components=96, n_social_features=15, random_state=42)),
            (
                "hgb",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.07,
                    max_depth=6,
                    min_samples_leaf=20,
                    l2_regularization=0.2,
                    random_state=42,
                ),
            ),
        ]
    )


def _build_rf_pipeline() -> Pipeline:
    """Pipeline SVD (chỉ phần TF-IDF) + 15 Social Features nguyên bản + RandomForest."""
    return Pipeline(
        [
            ("svd_soc", _SVDSocialFeatureExtractor(n_components=96, n_social_features=15, random_state=42)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=18,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def _build_rbf_svm_pipeline(max_samples: int = 3500) -> _SubsampledClassifier:
    """Pipeline SVD + Social Features + StandardScaler + Calibrated SVC(kernel='rbf')."""
    return _SubsampledClassifier(
        estimator=Pipeline(
            [
                ("svd_soc", _SVDSocialFeatureExtractor(n_components=96, n_social_features=15, random_state=42)),
                ("scaler", StandardScaler()),
                (
                    "svc",
                    CalibratedClassifierCV(
                        SVC(
                            kernel="rbf",
                            C=1.5,
                            gamma="scale",
                            class_weight="balanced",
                            random_state=42,
                        ),
                        cv=3,
                        ensemble=False,
                    ),
                ),
            ]
        ),
        max_samples=max_samples,
        random_state=42,
    )


def _tune_classifier(
    factory: Any,
    grid: Sequence[dict[str, Any]],
    X_tr_sub: Any,
    y_tr_sub: Sequence[str],
    X_val_sub: Any,
    y_val_sub: Sequence[str],
    default_params: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    """Tinh chỉnh siêu tham số trên validation split (tối ưu Macro-F1)."""
    best_params = dict(default_params)
    best_f1 = -1.0
    for params in grid:
        try:
            clf = factory(**params)
            clf.fit(X_tr_sub, y_tr_sub)
            preds = clf.predict(X_val_sub)
            score = float(f1_score(y_val_sub, preds, average="macro", zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_params = dict(params)
        except Exception:
            continue
    return best_params, round(max(0.0, best_f1), 4)


def _fit_oof_ensemble(
    base_factories: dict[str, Any],
    X_train: Any,
    y_train: Sequence[str],
    classes: Sequence[str],
    val_scores: dict[str, float],
    meta_pool_names: Sequence[str] | None = None,
) -> tuple[dict[str, Any], Any, str, dict[str, Any]]:
    """Huấn luyện toàn bộ mô hình cơ sở và chọn chiến lược Meta-Ensemble có OOF Macro-F1 cao nhất."""
    y_arr = np.asarray(list(y_train), dtype=object)
    counts = Counter(y_arr.tolist())
    min_count = min(counts.values()) if counts else 0
    all_names = list(base_factories.keys())
    pool_names = [n for n in (meta_pool_names or all_names) if n in base_factories]
    if not pool_names:
        pool_names = list(all_names)
    classes_list = list(classes)

    fitted_bases: dict[str, Any] = {}
    if min_count >= 3 and len(y_arr) >= 30:
        n_splits = 3
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_blocks: dict[str, np.ndarray] = {
            name: np.zeros((len(y_arr), len(classes_list)), dtype=np.float64)
            for name in pool_names
        }
        for tr_idx, va_idx in skf.split(np.zeros(len(y_arr)), y_arr):
            X_f_tr, X_f_va = X_train[tr_idx], X_train[va_idx]
            y_f_tr = y_arr[tr_idx]
            for name in pool_names:
                clf = base_factories[name]()
                clf.fit(X_f_tr, y_f_tr)
                oof_blocks[name][va_idx] = _OOFStackingEnsemble._extract_aligned_probs(
                    clf, X_f_va, classes_list
                )

        oof_f1_by_model: dict[str, float] = {}
        cls_arr = np.asarray(classes_list, dtype=object)
        for name in pool_names:
            oof_preds = cls_arr[np.argmax(oof_blocks[name], axis=1)]
            oof_f1_by_model[name] = round(
                float(f1_score(y_arr, oof_preds, average="macro", zero_division=0)),
                4,
            )

        ranked_models = sorted(pool_names, key=lambda n: oof_f1_by_model[n], reverse=True)
        top5_names = ranked_models[: min(5, len(ranked_models))]
        top4_names = ranked_models[: min(4, len(ranked_models))]

        meta_X_top5 = _OOFStackingEnsemble._probs_dict_to_meta_features(oof_blocks, top5_names)
        best_stack_params = {"C": 0.5, "class_weight": None}
        best_stack_f1 = -1.0
        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
        for cw in (None, "balanced"):
            for c_val in (0.05, 0.1, 0.5, 1.0):
                inner_preds = np.empty(len(y_arr), dtype=object)
                for itr, iva in inner_skf.split(np.zeros(len(y_arr)), y_arr):
                    m_clf = LogisticRegression(
                        C=c_val,
                        max_iter=1000,
                        class_weight=cw,
                        random_state=42,
                    )
                    m_clf.fit(meta_X_top5[itr], y_arr[itr])
                    inner_preds[iva] = m_clf.predict(meta_X_top5[iva])
                score_a = float(f1_score(y_arr, inner_preds, average="macro", zero_division=0))
                if score_a > best_stack_f1:
                    best_stack_f1 = score_a
                    best_stack_params = {"C": c_val, "class_weight": cw}

        weights_b = {m: round(max(0.05, oof_f1_by_model[m] - 0.30) ** 2, 4) for m in top4_names}
        prob_b = np.zeros((len(y_arr), len(classes_list)), dtype=np.float64)
        total_wb = sum(weights_b.values()) or 1.0
        for m in top4_names:
            prob_b += (weights_b[m] / total_wb) * oof_blocks[m]
        preds_b = cls_arr[np.argmax(prob_b, axis=1)]
        f1_b = float(f1_score(y_arr, preds_b, average="macro", zero_division=0))

        prob_c = sum(oof_blocks[m] for m in top4_names) / max(1, len(top4_names))
        preds_c = cls_arr[np.argmax(prob_c, axis=1)]
        f1_c = float(f1_score(y_arr, preds_c, average="macro", zero_division=0))

        for name, make_fn in base_factories.items():
            clf = make_fn()
            clf.fit(X_train, y_arr)
            fitted_bases[name] = clf

        candidates_f1 = {
            "oof_stacking_lr": round(best_stack_f1, 4),
            "top4_weighted_soft_vote": round(f1_b, 4),
            "top4_uniform_soft_vote": round(f1_c, 4),
        }
        best_method = max(candidates_f1, key=candidates_f1.get)  # type: ignore[arg-type]
        selection_details: dict[str, Any] = {
            "selected_method": best_method,
            "oof_strategy_macro_f1": candidates_f1,
            "oof_base_macro_f1": oof_f1_by_model,
            "top5_stacking_models": top5_names,
            "top4_vote_models": top4_names,
            "stacking_best_params": best_stack_params,
        }

        if best_method == "oof_stacking_lr":
            meta_clf = LogisticRegression(
                C=best_stack_params["C"],
                max_iter=1000,
                class_weight=best_stack_params["class_weight"],
                random_state=42,
            )
            meta_clf.fit(meta_X_top5, y_arr)
            ensemble: Any = _OOFStackingEnsemble(
                base_model_names=top5_names,
                classes=classes_list,
                base_models={m: fitted_bases[m] for m in top5_names},
                meta_model=meta_clf,
            )
        elif best_method == "top4_weighted_soft_vote":
            ensemble = _WeightedSoftVoteEnsemble(
                base_model_names=top4_names,
                weights=weights_b,
                classes=classes_list,
                base_models={m: fitted_bases[m] for m in top4_names},
            )
        else:
            ensemble = _WeightedSoftVoteEnsemble(
                base_model_names=top4_names,
                weights={m: 1.0 for m in top4_names},
                classes=classes_list,
                base_models={m: fitted_bases[m] for m in top4_names},
            )
        return fitted_bases, ensemble, best_method, selection_details

    for name, make_fn in base_factories.items():
        clf = make_fn()
        clf.fit(X_train, y_arr)
        fitted_bases[name] = clf

    ensemble_ws = _WeightedSoftVoteEnsemble(
        base_model_names=pool_names,
        weights=val_scores,
        classes=classes_list,
        base_models={m: fitted_bases[m] for m in pool_names},
    )
    return fitted_bases, ensemble_ws, "weighted_soft_vote", {"selected_method": "weighted_soft_vote"}


class _LanguageMLBundle:
    """Bộ mô hình ML cổ điển nâng cao v3 cho một ngôn ngữ ('vi' hoặc 'en')."""

    def __init__(self, language: str, cache_dir: Path) -> None:
        self.language = language
        self.cache_dir = cache_dir
        self.polarity_vectorizer: FeatureUnion | None = None
        self.polarity_models: dict[str, Any] = {}
        self.emotion_vectorizer: FeatureUnion | None = None
        self.emotion_models: dict[str, Any] = {}
        self.benchmark_metrics: dict[str, Any] = {}
        self.last_polarity_timings: dict[str, float] = {}
        self.last_emotion_timings: dict[str, float] = {}

    def _cache_paths(self) -> tuple[Path, Path, Path]:
        return (
            self.cache_dir / "polarity_models.joblib",
            self.cache_dir / "emotion_ml_model.joblib",
            self.cache_dir / "benchmark_metrics.json",
        )

    def load_or_train(self, force_retrain: bool = False, tune: bool = True) -> dict[str, Any]:
        pol_path, emo_path, bench_path = self._cache_paths()

        if (
            not force_retrain
            and pol_path.is_file()
            and emo_path.is_file()
            and bench_path.is_file()
        ):
            try:
                pol_bundle = joblib.load(pol_path)
                emo_bundle = joblib.load(emo_path)
                bench = json.loads(bench_path.read_text(encoding="utf-8"))
                if (
                    pol_bundle.get("schema_version") == ML_BUNDLE_SCHEMA_VERSION
                    and emo_bundle.get("schema_version") == ML_BUNDLE_SCHEMA_VERSION
                    and bench.get("schema_version") == ML_BUNDLE_SCHEMA_VERSION
                ):
                    self.polarity_vectorizer = pol_bundle["vectorizer"]
                    self.polarity_models = pol_bundle["models"]
                    self.emotion_vectorizer = emo_bundle["vectorizer"]
                    self.emotion_models = emo_bundle["models"]
                    self.benchmark_metrics = bench
                    return self.benchmark_metrics
            except Exception as exc:
                print(f"[Cảnh báo] Không nạp được cache ML ({self.language}): {exc}")

        print(
            f"[Classical ML v3 - {self.language.upper()}] Đang nạp dữ liệu & huấn luyện "
            "5 họ ML (NB/CNB, LR/Ridge/SGD, LinearSVC/NBSVM/RBF-SVM, RF/HistGB + Smart Ensemble)..."
        )
        if self.language == "en":
            (
                pol_train_texts,
                pol_train_labels,
                pol_test_texts,
                pol_test_labels,
                emo_train_texts,
                emo_train_labels,
                emo_test_texts,
                emo_test_labels,
            ) = _load_en_corpora()

            if len(pol_train_texts) < 100:
                pol_train_texts = [t for t, _ in _FALLBACK_EN_POLARITY_DATA] * 6
                pol_train_labels = [l for _, l in _FALLBACK_EN_POLARITY_DATA] * 6
                pol_test_texts = [t for t, _ in _FALLBACK_EN_POLARITY_DATA]
                pol_test_labels = [l for _, l in _FALLBACK_EN_POLARITY_DATA]
            else:
                pol_train_texts.extend([t for t, _ in _FALLBACK_EN_POLARITY_DATA] * 4)
                pol_train_labels.extend([l for _, l in _FALLBACK_EN_POLARITY_DATA] * 4)

            emo_train_texts.extend([t for t, _ in _SUPPLEMENT_EN_EMOTION_DATA] * 10)
            emo_train_labels.extend([l for _, l in _SUPPLEMENT_EN_EMOTION_DATA] * 10)
            if not emo_test_texts:
                emo_test_texts = [t for t, _ in _SUPPLEMENT_EN_EMOTION_DATA]
                emo_test_labels = [l for _, l in _SUPPLEMENT_EN_EMOTION_DATA]
            else:
                extra_eval = [
                    (t, l)
                    for t, l in _SUPPLEMENT_EN_EMOTION_DATA
                    if l in {"Disgust", "Other"}
                ]
                emo_test_texts.extend([t for t, _ in extra_eval])
                emo_test_labels.extend([l for _, l in extra_eval])

            pol_train_texts, pol_train_labels = _normalize_corpus_texts(
                pol_train_texts, pol_train_labels, language="en", segment_vi=False
            )
            pol_test_texts, pol_test_labels = _normalize_corpus_texts(
                pol_test_texts, pol_test_labels, language="en", segment_vi=False
            )
            emo_train_texts, emo_train_labels = _normalize_corpus_texts(
                emo_train_texts, emo_train_labels, language="en", segment_vi=False
            )
            emo_test_texts, emo_test_labels = _normalize_corpus_texts(
                emo_test_texts, emo_test_labels, language="en", segment_vi=False
            )
        else:
            (
                pol_train_texts,
                pol_train_labels,
                pol_test_texts,
                pol_test_labels,
                emo_train_texts,
                emo_train_labels,
                emo_test_texts,
                emo_test_labels,
            ) = _load_vi_corpora()

            if len(pol_train_texts) < 100:
                pol_train_texts = [t for t, _ in _FALLBACK_VI_POLARITY_DATA] * 5
                pol_train_labels = [l for _, l in _FALLBACK_VI_POLARITY_DATA] * 5
                pol_test_texts = [t for t, _ in _FALLBACK_VI_POLARITY_DATA]
                pol_test_labels = [l for _, l in _FALLBACK_VI_POLARITY_DATA]

            if len(emo_train_texts) < 100:
                emo_train_texts = [t for t, _ in _FALLBACK_VI_EMOTION_DATA] * 5
                emo_train_labels = [l for _, l in _FALLBACK_VI_EMOTION_DATA] * 5
                emo_test_texts = [t for t, _ in _FALLBACK_VI_EMOTION_DATA]
                emo_test_labels = [l for _, l in _FALLBACK_VI_EMOTION_DATA]

            pol_train_texts, pol_train_labels = _normalize_corpus_texts(
                pol_train_texts, pol_train_labels, language="vi", segment_vi=True
            )
            pol_test_texts, pol_test_labels = _normalize_corpus_texts(
                pol_test_texts, pol_test_labels, language="vi", segment_vi=True
            )
            emo_train_texts, emo_train_labels = _normalize_corpus_texts(
                emo_train_texts, emo_train_labels, language="vi", segment_vi=True
            )
            emo_test_texts, emo_test_labels = _normalize_corpus_texts(
                emo_test_texts, emo_test_labels, language="vi", segment_vi=True
            )

        # --- PHA A: POLARITY MODELS ---
        self.polarity_vectorizer = _build_feature_union(
            language=self.language, sample_count=len(pol_train_texts)
        )
        X_train = self.polarity_vectorizer.fit_transform(pol_train_texts)
        X_test = self.polarity_vectorizer.transform(pol_test_texts)

        pol_tuned_params: dict[str, dict[str, Any]] = {
            "tfidf_nb": {"alpha": 0.5},
            "tfidf_cnb": {"alpha": 0.5, "norm": False},
            "tfidf_lr": {"C": 1.0, "class_weight": None},
            "tfidf_ridge": {"alpha": 1.0, "class_weight": None},
            "tfidf_sgd": {"loss": "log_loss", "alpha": 5e-5, "class_weight": None},
            "tfidf_svm": {"C": 0.5, "class_weight": None},
            "tfidf_nbsvm": {"C": 0.5, "class_weight": None},
        }
        pol_val_scores: dict[str, float] = {k: 1.0 for k in POLARITY_MODEL_KEYS[:-1]}

        if tune and len(pol_train_labels) >= 60 and min(Counter(pol_train_labels).values()) >= 4:
            tr_i, va_i = train_test_split(
                np.arange(len(pol_train_labels)),
                test_size=0.15,
                random_state=42,
                stratify=pol_train_labels,
            )
            X_tr_sub, X_va_sub = X_train[tr_i], X_train[va_i]
            y_tr_sub = [pol_train_labels[i] for i in tr_i]
            y_va_sub = [pol_train_labels[i] for i in va_i]

            pol_tuned_params["tfidf_nb"], pol_val_scores["tfidf_nb"] = _tune_classifier(
                lambda **kw: MultinomialNB(**kw),
                [{"alpha": 0.2}, {"alpha": 0.5}, {"alpha": 1.0}],
                X_tr_sub,
                y_tr_sub,
                X_va_sub,
                y_va_sub,
                {"alpha": 0.5},
            )
            pol_tuned_params["tfidf_cnb"], pol_val_scores["tfidf_cnb"] = _tune_classifier(
                lambda **kw: ComplementNB(norm=False, **kw),
                [{"alpha": 0.2}, {"alpha": 0.5}, {"alpha": 1.0}],
                X_tr_sub,
                y_tr_sub,
                X_va_sub,
                y_va_sub,
                {"alpha": 0.5},
            )
            pol_tuned_params["tfidf_cnb"]["norm"] = False
            pol_tuned_params["tfidf_lr"], pol_val_scores["tfidf_lr"] = _tune_classifier(
                lambda **kw: LogisticRegression(max_iter=1500, random_state=42, **kw),
                [
                    {"C": c_v, "class_weight": cw}
                    for cw in (None, "balanced")
                    for c_v in (0.5, 1.0, 2.0)
                ],
                X_tr_sub,
                y_tr_sub,
                X_va_sub,
                y_va_sub,
                {"C": 1.0, "class_weight": None},
            )
            pol_tuned_params["tfidf_ridge"], pol_val_scores["tfidf_ridge"] = _tune_classifier(
                lambda **kw: RidgeClassifier(random_state=42, **kw),
                [
                    {"alpha": a_v, "class_weight": cw}
                    for cw in (None, "balanced")
                    for a_v in (0.5, 1.0, 2.0)
                ],
                X_tr_sub,
                y_tr_sub,
                X_va_sub,
                y_va_sub,
                {"alpha": 1.0, "class_weight": None},
            )
            pol_tuned_params["tfidf_sgd"], pol_val_scores["tfidf_sgd"] = _tune_classifier(
                lambda **kw: SGDClassifier(max_iter=1500, tol=1e-4, random_state=42, **kw),
                [
                    {"loss": ls, "alpha": a_v, "class_weight": cw}
                    for ls in ("log_loss", "modified_huber")
                    for cw in (None, "balanced")
                    for a_v in (1e-5, 5e-5)
                ],
                X_tr_sub,
                y_tr_sub,
                X_va_sub,
                y_va_sub,
                {"loss": "log_loss", "alpha": 5e-5, "class_weight": None},
            )
            pol_tuned_params["tfidf_svm"], pol_val_scores["tfidf_svm"] = _tune_classifier(
                lambda **kw: LinearSVC(max_iter=3000, random_state=42, **kw),
                [
                    {"C": c_v, "class_weight": cw}
                    for cw in (None, "balanced")
                    for c_v in (0.25, 0.5, 1.0)
                ],
                X_tr_sub,
                y_tr_sub,
                X_va_sub,
                y_va_sub,
                {"C": 0.5, "class_weight": None},
            )
            pol_tuned_params["tfidf_nbsvm"], pol_val_scores["tfidf_nbsvm"] = _tune_classifier(
                lambda **kw: _NBSVMClassifier(cv=2, random_state=42, **kw),
                [
                    {"C": 0.25, "class_weight": None},
                    {"C": 0.5, "class_weight": None},
                ],
                X_tr_sub,
                y_tr_sub,
                X_va_sub,
                y_va_sub,
                {"C": 0.5, "class_weight": None},
            )

        nb_p = dict(pol_tuned_params["tfidf_nb"])
        cnb_p = {**pol_tuned_params["tfidf_cnb"], "norm": False}
        lr_p = dict(pol_tuned_params["tfidf_lr"])
        ridge_p = dict(pol_tuned_params["tfidf_ridge"])
        sgd_p = dict(pol_tuned_params["tfidf_sgd"])
        svm_p = dict(pol_tuned_params["tfidf_svm"])
        nbsvm_p = dict(pol_tuned_params["tfidf_nbsvm"])

        pol_factories: dict[str, Any] = {
            "tfidf_nb": lambda: MultinomialNB(**nb_p),
            "tfidf_cnb": lambda: ComplementNB(**cnb_p),
            "tfidf_lr": lambda: LogisticRegression(max_iter=1500, random_state=42, **lr_p),
            "tfidf_ridge": lambda: CalibratedClassifierCV(
                RidgeClassifier(random_state=42, **ridge_p),
                cv=3,
                method="sigmoid",
            ),
            "tfidf_sgd": lambda: SGDClassifier(
                max_iter=1500, tol=1e-4, random_state=42, **sgd_p
            ),
            "tfidf_svm": lambda: CalibratedClassifierCV(
                LinearSVC(max_iter=3000, random_state=42, **svm_p),
                cv=3,
                method="sigmoid",
            ),
            "tfidf_nbsvm": lambda: _NBSVMClassifier(cv=2, random_state=42, **nbsvm_p),
            "tfidf_rbf_svm": lambda: _build_rbf_svm_pipeline(max_samples=3500),
            "tfidf_rf": _build_rf_pipeline,
            "tfidf_tree": _build_tree_pipeline,
        }

        pol_meta_pool = (
            "tfidf_nb",
            "tfidf_cnb",
            "tfidf_lr",
            "tfidf_ridge",
            "tfidf_sgd",
            "tfidf_svm",
            "tfidf_nbsvm",
        )
        fitted_pol, pol_ensemble, pol_ens_method, pol_sel_details = _fit_oof_ensemble(
            base_factories=pol_factories,
            X_train=X_train,
            y_train=pol_train_labels,
            classes=POLARITY_LABELS,
            val_scores=pol_val_scores,
            meta_pool_names=pol_meta_pool,
        )
        self.polarity_models = {**fitted_pol, "ml_ensemble": pol_ensemble}

        pol_bench: dict[str, dict[str, Any]] = {}
        for name, clf in self.polarity_models.items():
            preds = [str(p) for p in clf.predict(X_test)]
            metrics = _evaluate_predictions(pol_test_labels, preds, POLARITY_LABELS)
            if name in pol_tuned_params:
                metrics["best_params"] = pol_tuned_params[name]
            pol_bench[name] = metrics

        # --- PHA B: EMOTION MODELS ---
        e_train_x, e_test_x, e_train_y, e_test_y = (
            emo_train_texts,
            emo_test_texts,
            emo_train_labels,
            emo_test_labels,
        )
        if not e_test_x:
            e_train_x, e_test_x, e_train_y, e_test_y = train_test_split(
                emo_train_texts,
                emo_train_labels,
                test_size=0.15,
                random_state=42,
                stratify=emo_train_labels if len(set(emo_train_labels)) > 1 else None,
            )

        self.emotion_vectorizer = _build_feature_union(
            language=self.language, sample_count=len(e_train_x)
        )
        Xe_train = self.emotion_vectorizer.fit_transform(e_train_x)
        Xe_test = self.emotion_vectorizer.transform(e_test_x)

        emo_tuned_params: dict[str, dict[str, Any]] = {
            "emo_cnb": {"alpha": 0.5, "norm": False},
            "emo_lr": {"C": 2.0, "class_weight": None},
            "emo_ridge": {"alpha": 1.0, "class_weight": None},
            "emo_svm": {"C": 0.5, "class_weight": None},
            "emo_nbsvm": {"C": 0.5, "class_weight": None},
            "emo_sgd": {"loss": "log_loss", "alpha": 5e-5, "class_weight": None},
        }
        emo_val_scores: dict[str, float] = {k: 1.0 for k in EMOTION_MODEL_KEYS[:-1]}

        if tune and len(e_train_y) >= 50 and min(Counter(e_train_y).values()) >= 4:
            etr_i, eva_i = train_test_split(
                np.arange(len(e_train_y)),
                test_size=0.15,
                random_state=42,
                stratify=e_train_y,
            )
            Xe_tr_sub, Xe_va_sub = Xe_train[etr_i], Xe_train[eva_i]
            ye_tr_sub = [e_train_y[i] for i in etr_i]
            ye_va_sub = [e_train_y[i] for i in eva_i]

            emo_tuned_params["emo_cnb"], emo_val_scores["emo_cnb"] = _tune_classifier(
                lambda **kw: ComplementNB(norm=False, **kw),
                [{"alpha": 0.2}, {"alpha": 0.5}, {"alpha": 1.0}],
                Xe_tr_sub,
                ye_tr_sub,
                Xe_va_sub,
                ye_va_sub,
                {"alpha": 0.5},
            )
            emo_tuned_params["emo_cnb"]["norm"] = False
            emo_tuned_params["emo_lr"], emo_val_scores["emo_lr"] = _tune_classifier(
                lambda **kw: LogisticRegression(max_iter=1500, random_state=42, **kw),
                [
                    {"C": c_v, "class_weight": cw}
                    for cw in (None, "balanced")
                    for c_v in (0.5, 1.0, 2.0)
                ],
                Xe_tr_sub,
                ye_tr_sub,
                Xe_va_sub,
                ye_va_sub,
                {"C": 2.0, "class_weight": None},
            )
            emo_tuned_params["emo_ridge"], emo_val_scores["emo_ridge"] = _tune_classifier(
                lambda **kw: RidgeClassifier(random_state=42, **kw),
                [
                    {"alpha": a_v, "class_weight": cw}
                    for cw in (None, "balanced")
                    for a_v in (0.5, 1.0, 2.0)
                ],
                Xe_tr_sub,
                ye_tr_sub,
                Xe_va_sub,
                ye_va_sub,
                {"alpha": 1.0, "class_weight": None},
            )
            emo_tuned_params["emo_svm"], emo_val_scores["emo_svm"] = _tune_classifier(
                lambda **kw: LinearSVC(max_iter=3000, random_state=42, **kw),
                [
                    {"C": c_v, "class_weight": cw}
                    for cw in (None, "balanced")
                    for c_v in (0.25, 0.5, 1.0)
                ],
                Xe_tr_sub,
                ye_tr_sub,
                Xe_va_sub,
                ye_va_sub,
                {"C": 0.5, "class_weight": None},
            )
            emo_tuned_params["emo_nbsvm"], emo_val_scores["emo_nbsvm"] = _tune_classifier(
                lambda **kw: _NBSVMClassifier(cv=2, random_state=42, **kw),
                [
                    {"C": 0.25, "class_weight": None},
                    {"C": 0.5, "class_weight": None},
                ],
                Xe_tr_sub,
                ye_tr_sub,
                Xe_va_sub,
                ye_va_sub,
                {"C": 0.5, "class_weight": None},
            )
            emo_tuned_params["emo_sgd"], emo_val_scores["emo_sgd"] = _tune_classifier(
                lambda **kw: SGDClassifier(max_iter=1500, tol=1e-4, random_state=42, **kw),
                [
                    {"loss": ls, "alpha": a_v, "class_weight": cw}
                    for ls in ("log_loss", "modified_huber")
                    for cw in (None, "balanced")
                    for a_v in (1e-5, 5e-5)
                ],
                Xe_tr_sub,
                ye_tr_sub,
                Xe_va_sub,
                ye_va_sub,
                {"loss": "log_loss", "alpha": 5e-5, "class_weight": None},
            )

        ecnb_p = {**emo_tuned_params["emo_cnb"], "norm": False}
        elr_p = dict(emo_tuned_params["emo_lr"])
        eridge_p = dict(emo_tuned_params["emo_ridge"])
        esvm_p = dict(emo_tuned_params["emo_svm"])
        enbsvm_p = dict(emo_tuned_params["emo_nbsvm"])
        esgd_p = dict(emo_tuned_params["emo_sgd"])

        emo_factories: dict[str, Any] = {
            "emo_cnb": lambda: ComplementNB(**ecnb_p),
            "emo_lr": lambda: LogisticRegression(max_iter=1500, random_state=42, **elr_p),
            "emo_ridge": lambda: CalibratedClassifierCV(
                RidgeClassifier(random_state=42, **eridge_p),
                cv=3,
                method="sigmoid",
            ),
            "emo_svm": lambda: CalibratedClassifierCV(
                LinearSVC(max_iter=3000, random_state=42, **esvm_p),
                cv=3,
                method="sigmoid",
            ),
            "emo_nbsvm": lambda: _NBSVMClassifier(cv=2, random_state=42, **enbsvm_p),
            "emo_sgd": lambda: SGDClassifier(
                max_iter=1500, tol=1e-4, random_state=42, **esgd_p
            ),
        }

        fitted_emo, emo_ensemble, emo_ens_method, emo_sel_details = _fit_oof_ensemble(
            base_factories=emo_factories,
            X_train=Xe_train,
            y_train=e_train_y,
            classes=EMOTION_LABELS,
            val_scores=emo_val_scores,
        )
        self.emotion_models = {**fitted_emo, "emo_ensemble": emo_ensemble}

        emo_bench: dict[str, dict[str, Any]] = {}
        for name, clf in self.emotion_models.items():
            preds = [str(p) for p in clf.predict(Xe_test)]
            metrics = _evaluate_predictions(e_test_y, preds, EMOTION_LABELS)
            if name in emo_tuned_params:
                metrics["best_params"] = emo_tuned_params[name]
            emo_bench[name] = metrics

        self.benchmark_metrics = {
            "schema_version": ML_BUNDLE_SCHEMA_VERSION,
            "language": self.language,
            "feature_pipeline": [
                "word_tfidf(1-3)",
                "char_wb_tfidf(3-5)",
                "social_lexicon_features(15)",
            ],
            "polarity_train_size": len(pol_train_texts),
            "polarity_test_size": len(pol_test_texts),
            "polarity_ensemble_method": pol_ens_method,
            "polarity_ensemble_selection": pol_sel_details,
            "polarity_models": pol_bench,
            "emotion_train_size": len(e_train_x),
            "emotion_test_size": len(e_test_x),
            "emotion_ensemble_method": emo_ens_method,
            "emotion_ensemble_selection": emo_sel_details,
            "emotion_models": emo_bench,
            "top_features": {
                "polarity": _extract_top_features(
                    self.polarity_vectorizer, self.polarity_models["tfidf_lr"], top_k=10
                ),
                "emotion": _extract_top_features(
                    self.emotion_vectorizer, self.emotion_models["emo_lr"], top_k=10
                ),
            },
        }

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "schema_version": ML_BUNDLE_SCHEMA_VERSION,
                "vectorizer": self.polarity_vectorizer,
                "models": self.polarity_models,
            },
            pol_path,
            compress=3,
        )
        joblib.dump(
            {
                "schema_version": ML_BUNDLE_SCHEMA_VERSION,
                "vectorizer": self.emotion_vectorizer,
                "models": self.emotion_models,
            },
            emo_path,
            compress=3,
        )
        bench_path.write_text(
            json.dumps(self.benchmark_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.benchmark_metrics

    def ensure_loaded(self) -> None:
        if (
            self.polarity_vectorizer is None
            or not self.polarity_models
            or self.emotion_vectorizer is None
            or not self.emotion_models
        ):
            self.load_or_train(force_retrain=False)

    def predict_polarity_all(
        self,
        texts: Sequence[str],
        full_models: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        self.ensure_loaded()
        assert self.polarity_vectorizer is not None
        self.last_polarity_timings = {}
        if not texts:
            return {}

        t_vec0 = time.perf_counter()
        X = self.polarity_vectorizer.transform([str(t) for t in texts])
        vec_sec = time.perf_counter() - t_vec0
        outputs: dict[str, list[dict[str, Any]]] = {}

        all_base_names = [
            k for k in POLARITY_MODEL_KEYS if k != "ml_ensemble" and k in self.polarity_models
        ]
        ens = self.polarity_models.get("ml_ensemble")
        if full_models:
            base_names = all_base_names
        else:
            ens_deps = set(getattr(ens, "base_model_names", ()) or ()) if ens is not None else set()
            needed = (set(CORE_POLARITY_MODEL_KEYS) - {"ml_ensemble"}) | ens_deps
            base_names = [k for k in all_base_names if k in needed]

        for name in base_names:
            t_m0 = time.perf_counter()
            clf = self.polarity_models[name]
            probs_mat = clf.predict_proba(X)
            classes = [str(c) for c in clf.classes_]
            preds: list[dict[str, Any]] = []
            for row in probs_mat:
                score_map = {"POS": 0.0, "NEG": 0.0, "NEU": 0.0}
                for cls_name, p in zip(classes, row):
                    if cls_name in score_map:
                        score_map[cls_name] = round(float(p), 4)
                best_lbl = max(score_map, key=score_map.get)  # type: ignore[arg-type]
                preds.append(
                    {"label": best_lbl, "confidence": score_map[best_lbl], "scores": score_map}
                )
            outputs[name] = preds
            self.last_polarity_timings[name] = round(time.perf_counter() - t_m0, 4)

        if ens is not None:
            t_ens0 = time.perf_counter()
            if hasattr(ens, "predict_proba_from_base_probs") and all(
                m in outputs for m in getattr(ens, "base_model_names", ())
            ):
                probs_mat = ens.predict_proba_from_base_probs(outputs)
            else:
                probs_mat = ens.predict_proba(X)
            classes = [str(c) for c in ens.classes_]
            ens_preds: list[dict[str, Any]] = []
            for row in probs_mat:
                score_map = {"POS": 0.0, "NEG": 0.0, "NEU": 0.0}
                for cls_name, p in zip(classes, row):
                    if cls_name in score_map:
                        score_map[cls_name] = round(float(p), 4)
                best_lbl = max(score_map, key=score_map.get)  # type: ignore[arg-type]
                ens_preds.append(
                    {"label": best_lbl, "confidence": score_map[best_lbl], "scores": score_map}
                )
            outputs["ml_ensemble"] = ens_preds
            self.last_polarity_timings["ml_ensemble"] = round(time.perf_counter() - t_ens0, 4)

        self.last_polarity_timings["_vectorizer"] = round(vec_sec, 4)
        return outputs

    def predict_emotion_all(
        self,
        texts: Sequence[str],
        full_models: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        self.ensure_loaded()
        assert self.emotion_vectorizer is not None
        self.last_emotion_timings = {}
        if not texts:
            return {}

        t_vec0 = time.perf_counter()
        X = self.emotion_vectorizer.transform([str(t) for t in texts])
        vec_sec = time.perf_counter() - t_vec0
        outputs: dict[str, list[dict[str, Any]]] = {}

        all_base_names = [
            k for k in EMOTION_MODEL_KEYS if k != "emo_ensemble" and k in self.emotion_models
        ]
        ens = self.emotion_models.get("emo_ensemble")
        if full_models:
            base_names = all_base_names
        else:
            ens_deps = set(getattr(ens, "base_model_names", ()) or ()) if ens is not None else set()
            needed = (set(CORE_EMOTION_MODEL_KEYS) - {"emo_ensemble"}) | ens_deps
            base_names = [k for k in all_base_names if k in needed]

        for name in base_names:
            t_m0 = time.perf_counter()
            clf = self.emotion_models[name]
            probs_mat = clf.predict_proba(X)
            classes = [str(c) for c in clf.classes_]
            preds: list[dict[str, Any]] = []
            for row in probs_mat:
                score_map = {lbl: 0.0 for lbl in EMOTION_LABELS}
                for cls_name, p in zip(classes, row):
                    if cls_name in score_map:
                        score_map[cls_name] = round(float(p), 4)
                best_lbl = max(score_map, key=score_map.get)  # type: ignore[arg-type]
                preds.append(
                    {
                        "label": best_lbl,
                        "label_vi": EMOTION_VI_MAP.get(best_lbl, best_lbl),
                        "confidence": score_map[best_lbl],
                        "scores": score_map,
                    }
                )
            outputs[name] = preds
            self.last_emotion_timings[name] = round(time.perf_counter() - t_m0, 4)

        if ens is not None:
            t_ens0 = time.perf_counter()
            if hasattr(ens, "predict_proba_from_base_probs") and all(
                m in outputs for m in getattr(ens, "base_model_names", ())
            ):
                probs_mat = ens.predict_proba_from_base_probs(outputs)
            else:
                probs_mat = ens.predict_proba(X)
            classes = [str(c) for c in ens.classes_]
            ens_preds: list[dict[str, Any]] = []
            for row in probs_mat:
                score_map = {lbl: 0.0 for lbl in EMOTION_LABELS}
                for cls_name, p in zip(classes, row):
                    if cls_name in score_map:
                        score_map[cls_name] = round(float(p), 4)
                best_lbl = max(score_map, key=score_map.get)  # type: ignore[arg-type]
                ens_preds.append(
                    {
                        "label": best_lbl,
                        "label_vi": EMOTION_VI_MAP.get(best_lbl, best_lbl),
                        "confidence": score_map[best_lbl],
                        "scores": score_map,
                    }
                )
            outputs["emo_ensemble"] = ens_preds
            self.last_emotion_timings["emo_ensemble"] = round(time.perf_counter() - t_ens0, 4)

        self.last_emotion_timings["_vectorizer"] = round(vec_sec, 4)
        return outputs


class ClassicalMLSuite:
    """Bộ mô hình ML cổ điển song ngữ (VI & EN) cho Polarity (10+1) và Emotion (6+1)."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.bundles: dict[str, _LanguageMLBundle] = {
            "vi": _LanguageMLBundle("vi", self.cache_dir / "vi"),
            "en": _LanguageMLBundle("en", self.cache_dir / "en"),
        }
        self.benchmark_metrics: dict[str, Any] = {}
        self.last_polarity_timings: dict[str, float] = {}
        self.last_emotion_timings: dict[str, float] = {}

    def load_or_train(
        self,
        language: str = "vi",
        force_retrain: bool = False,
        tune: bool = True,
    ) -> dict[str, Any]:
        lang = "en" if language == "en" else "vi"
        metrics = self.bundles[lang].load_or_train(force_retrain=force_retrain, tune=tune)
        self.benchmark_metrics[lang] = metrics
        return metrics

    def ensure_loaded(self, language: str = "vi") -> None:
        lang = "en" if language == "en" else "vi"
        self.bundles[lang].ensure_loaded()
        self.benchmark_metrics[lang] = self.bundles[lang].benchmark_metrics

    def predict_polarity_all(
        self,
        segmented_texts: Sequence[str],
        languages: Sequence[str] | None = None,
        full_models: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Định tuyến từng comment qua cụm mô hình ML cổ điển đúng ngôn ngữ ('vi' hoặc 'en')."""
        self.last_polarity_timings = {}
        n = len(segmented_texts)
        if n == 0:
            return {}

        langs = list(languages) if languages is not None else ["vi"] * n
        vi_idx = [i for i, l in enumerate(langs) if l != "en"]
        en_idx = [i for i, l in enumerate(langs) if l == "en"]

        merged: dict[str, list[dict[str, Any] | None]] = {}

        def _merge_branch(
            indices: list[int],
            branch_res: dict[str, list[dict[str, Any]]],
            branch_timings: dict[str, float],
        ) -> None:
            for m_name, items in branch_res.items():
                if m_name not in merged:
                    merged[m_name] = [None] * n
                for orig_i, pred in zip(indices, items):
                    merged[m_name][orig_i] = pred
            for m_name, t_val in branch_timings.items():
                self.last_polarity_timings[m_name] = round(
                    self.last_polarity_timings.get(m_name, 0.0) + float(t_val), 4
                )

        if vi_idx:
            self.ensure_loaded("vi")
            vi_res = self.bundles["vi"].predict_polarity_all(
                [segmented_texts[i] for i in vi_idx],
                full_models=full_models,
            )
            _merge_branch(vi_idx, vi_res, self.bundles["vi"].last_polarity_timings)

        if en_idx:
            self.ensure_loaded("en")
            en_res = self.bundles["en"].predict_polarity_all(
                [segmented_texts[i] for i in en_idx],
                full_models=full_models,
            )
            _merge_branch(en_idx, en_res, self.bundles["en"].last_polarity_timings)

        # Chỉ giữ các mô hình thực sự chạy đủ trên tất cả mẫu hợp lệ (hoặc lấp rỗng nếu lệch giữa VI/EN)
        out: dict[str, list[dict[str, Any]]] = {}
        for k, v in merged.items():
            if all(item is not None for item in v):
                out[k] = [item for item in v if item is not None]
        return out

    def predict_emotion_all(
        self,
        segmented_texts: Sequence[str],
        languages: Sequence[str] | None = None,
        full_models: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        self.last_emotion_timings = {}
        n = len(segmented_texts)
        if n == 0:
            return {}
        langs = list(languages) if languages is not None else ["vi"] * n
        vi_idx = [i for i, l in enumerate(langs) if l != "en"]
        en_idx = [i for i, l in enumerate(langs) if l == "en"]

        merged: dict[str, list[dict[str, Any] | None]] = {}

        def _merge_branch(
            indices: list[int],
            branch_res: dict[str, list[dict[str, Any]]],
            branch_timings: dict[str, float],
        ) -> None:
            for m_name, items in branch_res.items():
                if m_name not in merged:
                    merged[m_name] = [None] * n
                for orig_i, pred in zip(indices, items):
                    merged[m_name][orig_i] = pred
            for m_name, t_val in branch_timings.items():
                self.last_emotion_timings[m_name] = round(
                    self.last_emotion_timings.get(m_name, 0.0) + float(t_val), 4
                )

        if vi_idx:
            self.ensure_loaded("vi")
            vi_res = self.bundles["vi"].predict_emotion_all(
                [segmented_texts[i] for i in vi_idx],
                full_models=full_models,
            )
            _merge_branch(vi_idx, vi_res, self.bundles["vi"].last_emotion_timings)

        if en_idx:
            self.ensure_loaded("en")
            en_res = self.bundles["en"].predict_emotion_all(
                [segmented_texts[i] for i in en_idx],
                full_models=full_models,
            )
            _merge_branch(en_idx, en_res, self.bundles["en"].last_emotion_timings)

        out: dict[str, list[dict[str, Any]]] = {}
        for k, v in merged.items():
            if all(item is not None for item in v):
                out[k] = [item for item in v if item is not None]
        return out


def _print_benchmark_report(lang: str, metrics: dict[str, Any]) -> None:
    print(f"\n=== BENCHMARK CLASSICAL ML v3 ({lang.upper()}) ===")
    print(f"- Feature Pipeline: {' + '.join(metrics.get('feature_pipeline', []))}")
    print(
        f"- Polarity ({metrics.get('polarity_train_size')} train / {metrics.get('polarity_test_size')} test) "
        f"| Ensemble: {metrics.get('polarity_ensemble_method')}"
    )
    for name, m in (metrics.get("polarity_models") or {}).items():
        print(
            f"  * {name:14s} | Acc={m['accuracy']:.4f} | Macro-F1={m['macro_f1']:.4f} | "
            f"Weighted-F1={m['weighted_f1']:.4f}"
        )
    print(
        f"- Emotion  ({metrics.get('emotion_train_size')} train / {metrics.get('emotion_test_size')} test) "
        f"| Ensemble: {metrics.get('emotion_ensemble_method')}"
    )
    for name, m in (metrics.get("emotion_models") or {}).items():
        print(
            f"  * {name:14s} | Acc={m['accuracy']:.4f} | Macro-F1={m['macro_f1']:.4f} | "
            f"Weighted-F1={m['weighted_f1']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Huấn luyện và benchmark bộ mô hình Classical ML v3 song ngữ (VI & EN)."
    )
    parser.add_argument(
        "--lang",
        nargs="+",
        choices=["vi", "en"],
        default=["vi", "en"],
        help="Ngôn ngữ cần nạp hoặc huấn luyện lại.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Bỏ qua cache cũ và huấn luyện lại từ đầu.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Tắt bước validation tuning để huấn luyện nhanh với tham số mặc định.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Thư mục lưu model cache và benchmark_metrics.json.",
    )
    args = parser.parse_args()

    suite = ClassicalMLSuite(cache_dir=args.cache_dir)
    for lang in args.lang:
        metrics = suite.load_or_train(
            language=lang,
            force_retrain=args.retrain,
            tune=not args.no_tune,
        )
        _print_benchmark_report(lang, metrics)


if __name__ == "__main__":
    from src.models.classical_ml import main as _module_main

    _module_main()
