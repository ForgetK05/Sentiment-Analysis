from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

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


# Alias tương thích ngược
VietnameseLexiconScorer = BilingualLexiconScorer

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
    ("i am so sad crying miss him heartbroken", "Sadness"),
    ("so disappointed lonely depressing makes me cry", "Sadness"),
    ("i am so fucking scared terrified creepy afraid", "Fear"),
    ("this gives me anxiety so nervous worried fear", "Fear"),
    ("so cute peak edit fire smooth love this amazing", "Enjoyment"),
    ("holy shit this is incredible so freaking good happy", "Enjoyment"),
    ("disgusting trash cringe pathetic gross awful", "Disgust"),
    ("ew hate this fake garbage makes me sick", "Disgust"),
    ("no way what is this i am witnessing omg shocked", "Surprise"),
    ("unbelievable plot twist unexpected surprise", "Surprise"),
    ("different mask same task see you in december", "Other"),
    ("what is the name of this song or movie", "Other"),
]


def _fast_segment_batch(texts: Sequence[str]) -> list[str]:
    try:
        from underthesea import word_tokenize

        return [word_tokenize(str(t), format="text") for t in texts]
    except Exception:
        return [str(t) for t in texts]


def _build_tfidf_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    token_pattern=r"(?u)\b[\w_]{1,}\b",
                    ngram_range=(1, 2),
                    max_features=25000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=25000,
                    sublinear_tf=True,
                ),
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


def _load_en_corpora() -> tuple[
    list[str], list[str], list[str], list[str], list[str], list[str], list[str], list[str]
]:
    """Tải plain-text TweetEval (sentiment & emotion) từ GitHub (không cần pyarrow)."""
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


class _LanguageMLBundle:
    """Bộ mô hình ML cổ điển cho một ngôn ngữ ('vi' hoặc 'en')."""

    def __init__(self, language: str, cache_dir: Path) -> None:
        self.language = language
        self.cache_dir = cache_dir
        self.polarity_vectorizer: FeatureUnion | None = None
        self.polarity_models: dict[str, Any] = {}
        self.emotion_vectorizer: FeatureUnion | None = None
        self.emotion_model: LogisticRegression | None = None
        self.benchmark_metrics: dict[str, Any] = {}

    def _cache_paths(self) -> tuple[Path, Path, Path]:
        return (
            self.cache_dir / "polarity_models.joblib",
            self.cache_dir / "emotion_ml_model.joblib",
            self.cache_dir / "benchmark_metrics.json",
        )

    def load_or_train(self, force_retrain: bool = False) -> dict[str, Any]:
        pol_path, emo_path, bench_path = self._cache_paths()

        if (
            not force_retrain
            and pol_path.is_file()
            and emo_path.is_file()
            and bench_path.is_file()
        ):
            pol_bundle = joblib.load(pol_path)
            emo_bundle = joblib.load(emo_path)
            self.polarity_vectorizer = pol_bundle["vectorizer"]
            self.polarity_models = pol_bundle["models"]
            self.emotion_vectorizer = emo_bundle["vectorizer"]
            self.emotion_model = emo_bundle["model"]
            self.benchmark_metrics = json.loads(bench_path.read_text(encoding="utf-8"))
            return self.benchmark_metrics

        print(f"[Classical ML - {self.language.upper()}] Đang nạp dữ liệu để huấn luyện NB, LR, SVM...")
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
                pol_train_texts = [t for t, _ in _FALLBACK_EN_POLARITY_DATA] * 5
                pol_train_labels = [l for _, l in _FALLBACK_EN_POLARITY_DATA] * 5
                pol_test_texts = [t for t, _ in _FALLBACK_EN_POLARITY_DATA]
                pol_test_labels = [l for _, l in _FALLBACK_EN_POLARITY_DATA]
            else:
                # Bổ sung ngữ cảnh TikTok (peak/fire/smooth/scared) vào tập train EN
                pol_train_texts.extend([t for t, _ in _FALLBACK_EN_POLARITY_DATA] * 3)
                pol_train_labels.extend([l for _, l in _FALLBACK_EN_POLARITY_DATA] * 3)

            # Bổ sung đủ 7 lớp cảm xúc cho bộ EN
            emo_train_texts.extend([t for t, _ in _SUPPLEMENT_EN_EMOTION_DATA] * 12)
            emo_train_labels.extend([l for _, l in _SUPPLEMENT_EN_EMOTION_DATA] * 12)
            if not emo_test_texts:
                emo_test_texts = [t for t, _ in _SUPPLEMENT_EN_EMOTION_DATA]
                emo_test_labels = [l for _, l in _SUPPLEMENT_EN_EMOTION_DATA]
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
                pol_train_texts = [t for t, _ in _FALLBACK_VI_POLARITY_DATA] * 4
                pol_train_labels = [l for _, l in _FALLBACK_VI_POLARITY_DATA] * 4
                pol_test_texts = [t for t, _ in _FALLBACK_VI_POLARITY_DATA]
                pol_test_labels = [l for _, l in _FALLBACK_VI_POLARITY_DATA]
            else:
                pol_train_texts = _fast_segment_batch(pol_train_texts)
                pol_test_texts = _fast_segment_batch(pol_test_texts)

            if len(emo_train_texts) < 100:
                emo_train_texts = [t for t, _ in _FALLBACK_VI_EMOTION_DATA] * 4
                emo_train_labels = [l for _, l in _FALLBACK_VI_EMOTION_DATA] * 4
                emo_test_texts = [t for t, _ in _FALLBACK_VI_EMOTION_DATA]
                emo_test_labels = [l for _, l in _FALLBACK_VI_EMOTION_DATA]
            else:
                emo_train_texts = _fast_segment_batch(emo_train_texts)
                emo_test_texts = _fast_segment_batch(emo_test_texts)

        self.polarity_vectorizer = _build_tfidf_vectorizer()
        X_train = self.polarity_vectorizer.fit_transform(pol_train_texts)
        X_test = self.polarity_vectorizer.transform(pol_test_texts)

        candidates = {
            "tfidf_nb": MultinomialNB(alpha=0.5),
            "tfidf_lr": LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced"),
            "tfidf_svm": CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced", random_state=42),
                cv=3,
            ),
        }

        pol_bench: dict[str, dict[str, float]] = {}
        self.polarity_models = {}
        for name, clf in candidates.items():
            clf.fit(X_train, pol_train_labels)
            preds = clf.predict(X_test)
            pol_bench[name] = {
                "accuracy": round(float(accuracy_score(pol_test_labels, preds)), 4),
                "macro_f1": round(
                    float(f1_score(pol_test_labels, preds, average="macro", zero_division=0)),
                    4,
                ),
            }
            self.polarity_models[name] = clf

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

        self.emotion_vectorizer = _build_tfidf_vectorizer()
        Xe_train = self.emotion_vectorizer.fit_transform(e_train_x)
        Xe_test = self.emotion_vectorizer.transform(e_test_x)
        self.emotion_model = LogisticRegression(
            C=2.0,
            max_iter=1000,
            class_weight="balanced",
        )
        self.emotion_model.fit(Xe_train, e_train_y)
        emo_preds = self.emotion_model.predict(Xe_test)

        self.benchmark_metrics = {
            "language": self.language,
            "polarity_train_size": len(pol_train_texts),
            "polarity_test_size": len(pol_test_texts),
            "polarity_models": pol_bench,
            "emotion_train_size": len(e_train_x),
            "emotion_test_size": len(e_test_x),
            "emotion_lr": {
                "accuracy": round(float(accuracy_score(e_test_y, emo_preds)), 4),
                "macro_f1": round(
                    float(f1_score(e_test_y, emo_preds, average="macro", zero_division=0)),
                    4,
                ),
            },
        }

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"vectorizer": self.polarity_vectorizer, "models": self.polarity_models},
            pol_path,
        )
        joblib.dump(
            {"vectorizer": self.emotion_vectorizer, "model": self.emotion_model},
            emo_path,
        )
        bench_path.write_text(
            json.dumps(self.benchmark_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.benchmark_metrics

    def ensure_loaded(self) -> None:
        if self.polarity_vectorizer is None or not self.polarity_models or self.emotion_model is None:
            self.load_or_train(force_retrain=False)

    def predict_polarity_all(self, texts: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
        self.ensure_loaded()
        assert self.polarity_vectorizer is not None
        if not texts:
            return {name: [] for name in self.polarity_models}
        X = self.polarity_vectorizer.transform([str(t) for t in texts])
        outputs: dict[str, list[dict[str, Any]]] = {}
        for name, clf in self.polarity_models.items():
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
        return outputs

    def predict_emotion(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        self.ensure_loaded()
        assert self.emotion_vectorizer is not None
        assert self.emotion_model is not None
        if not texts:
            return []
        X = self.emotion_vectorizer.transform([str(t) for t in texts])
        probs_mat = self.emotion_model.predict_proba(X)
        classes = [str(c) for c in self.emotion_model.classes_]
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
        return preds


class ClassicalMLSuite:
    """Bộ mô hình ML cổ điển song ngữ (VI & EN) cho Polarity (NB, LR, SVM) và Emotion (LR)."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.bundles: dict[str, _LanguageMLBundle] = {
            "vi": _LanguageMLBundle("vi", self.cache_dir / "vi"),
            "en": _LanguageMLBundle("en", self.cache_dir / "en"),
        }
        self.benchmark_metrics: dict[str, Any] = {}

    def load_or_train(
        self,
        language: str = "vi",
        force_retrain: bool = False,
    ) -> dict[str, Any]:
        lang = "en" if language == "en" else "vi"
        metrics = self.bundles[lang].load_or_train(force_retrain=force_retrain)
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
    ) -> dict[str, list[dict[str, Any]]]:
        """Định tuyến từng comment qua mô hình ML cổ điển đúng ngôn ngữ ('vi' hoặc 'en')."""
        n = len(segmented_texts)
        if n == 0:
            return {"tfidf_nb": [], "tfidf_lr": [], "tfidf_svm": []}

        langs = list(languages) if languages is not None else ["vi"] * n
        vi_idx = [i for i, l in enumerate(langs) if l != "en"]
        en_idx = [i for i, l in enumerate(langs) if l == "en"]

        merged: dict[str, list[dict[str, Any] | None]] = {
            "tfidf_nb": [None] * n,
            "tfidf_lr": [None] * n,
            "tfidf_svm": [None] * n,
        }

        if vi_idx:
            self.ensure_loaded("vi")
            vi_res = self.bundles["vi"].predict_polarity_all([segmented_texts[i] for i in vi_idx])
            for m_name, items in vi_res.items():
                for orig_i, pred in zip(vi_idx, items):
                    merged[m_name][orig_i] = pred

        if en_idx:
            self.ensure_loaded("en")
            en_res = self.bundles["en"].predict_polarity_all([segmented_texts[i] for i in en_idx])
            for m_name, items in en_res.items():
                for orig_i, pred in zip(en_idx, items):
                    merged[m_name][orig_i] = pred

        return {
            k: [item for item in v if item is not None]
            for k, v in merged.items()
        }

    def predict_emotion(
        self,
        segmented_texts: Sequence[str],
        languages: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        n = len(segmented_texts)
        if n == 0:
            return []
        langs = list(languages) if languages is not None else ["vi"] * n
        vi_idx = [i for i, l in enumerate(langs) if l != "en"]
        en_idx = [i for i, l in enumerate(langs) if l == "en"]
        out: list[dict[str, Any] | None] = [None] * n

        if vi_idx:
            self.ensure_loaded("vi")
            vi_res = self.bundles["vi"].predict_emotion([segmented_texts[i] for i in vi_idx])
            for orig_i, pred in zip(vi_idx, vi_res):
                out[orig_i] = pred

        if en_idx:
            self.ensure_loaded("en")
            en_res = self.bundles["en"].predict_emotion([segmented_texts[i] for i in en_idx])
            for orig_i, pred in zip(en_idx, en_res):
                out[orig_i] = pred

        return [item for item in out if item is not None]
