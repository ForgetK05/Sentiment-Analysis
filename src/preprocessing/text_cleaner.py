from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

# Chuẩn hóa biểu tượng cảm xúc dạng chữ của TikTok sang emoji Unicode chuẩn
TIKTOK_TEXT_EMOJI_MAP: dict[str, str] = {
    "[cười chảy nước mắt]": "😂",
    "[cười to tán thành]": "😂",
    "[cười lớn]": "😆",
    "[cười]": "😄",
    "[cười nhếch mép]": "😏",
    "[cười thầm]": "🤭",
    "[khóc]": "😭",
    "[khóc lớn]": "😭",
    "[buồn]": "😢",
    "[tức giận]": "😠",
    "[giận dữ]": "😠",
    "[nổi giận]": "😡",
    "[sợ]": "😱",
    "[hoảng sợ]": "😨",
    "[ngạc nhiên]": "😮",
    "[sửng sốt]": "😲",
    "[tim]": "❤️",
    "[yêu]": "❤️",
    "[thích]": "👍",
    "[vỗ tay]": "👏",
    "[tuyệt vời]": "🤩",
    "[chán nản]": "😒",
    "[im lặng]": "🤫",
}

# Chuẩn hóa teencode tiếng Việt an toàn (bỏ các từ đa nghĩa "bt", "v", "t")
TEENCODE_MAP: dict[str, str] = {
    "ko": "không",
    "k": "không",
    "kh": "không",
    "hk": "không",
    "hok": "không",
    "kg": "không",
    "khum": "không",
    "hong": "không",
    "hông": "không",
    "hem": "không",
    "hổng": "không",
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
    "bth": "bình thường",
    "ghéc": "ghét",
    "get": "ghét",
    "ngta": "người ta",
    "nhìu": "nhiều",
    "wa": "quá",
    "wá": "quá",
    "iu": "yêu",
    "trc": "trước",
    "nx": "nữa",
    "lun": "luôn",
    "thui": "thôi",
    "zay": "vậy",
    "zậy": "vậy",
    "oke": "ok",
    "okela": "ok",
    "j": "gì",
}

_TIKTOK_EMOJI_PATTERN = re.compile(
    "|".join(re.escape(k) for k in sorted(TIKTOK_TEXT_EMOJI_MAP, key=len, reverse=True)),
    flags=re.IGNORECASE,
)
_TEENCODE_PATTERN = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(TEENCODE_MAP, key=len, reverse=True)) + r")(?!\w)",
    flags=re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|vt\.tiktok\.com/\S+", flags=re.IGNORECASE)
_MENTION_PATTERN = re.compile(r"@\S+")
_HASHTAG_PATTERN = re.compile(r"#[^\s#]+")
_REPEAT_CHAR_PATTERN = re.compile(r"(.)\1{3,}")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_ALPHA_PATTERN = re.compile(r"[^\W\d_]", flags=re.UNICODE)
_PURE_PUNCT_OR_EMOJI_PATTERN = re.compile(r"^[\W\d_]+$", flags=re.UNICODE)

# Ký tự thuần Việt không xuất hiện trong Tây Ban Nha / Bồ Đào Nha / Pháp
_VI_UNIQUE_DIACRITIC_PATTERN = re.compile(
    r"[ăđơưảạằắẳẵặầấẩẫậẻẽẹềếểễệỉĩịỏọồốổỗộờớởỡợủũụừứửữựỳýỷỹỵ]",
    flags=re.IGNORECASE,
)
# Tất cả dấu tiếng Việt (gồm cả â, ê, ô, à, á, è, é, ì, í, ò, ó, ù, ú, ã, õ)
_VI_DIACRITIC_PATTERN = re.compile(
    r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]",
    flags=re.IGNORECASE,
)
_OTHER_LATIN_UNIQUE_CHARS = re.compile(r"[ñ¿¡çßäöü]", flags=re.IGNORECASE)

# Chữ cái phi-Latin (Trung, Nhật, Hàn, Nga/Cyrillic, Ả Rập, Thái, Ấn, Hy Lạp, Hebrew...)
_NON_LATIN_SCRIPT_PATTERN = re.compile(
    r"[\u0370-\u03FF\u0400-\u052F\u0590-\u05FF\u0600-\u06FF\u0900-\u097F"
    r"\u0E00-\u0E7F\u1100-\u11FF\u3040-\u30FF\u3130-\u318F\u4E00-\u9FFF\uAC00-\uD7AF]"
)

_FRIEND_CALL_WORDS = {
    "ơi", "oi", "nè", "ne", "kìa", "kia", "đây", "day", "vào", "vao",
    "xem", "coi", "đi", "di", "này", "nay", "ê", "nhé", "nhe", "nha",
    "hả", "ha", "đó", "do", "bạn", "ban", "mày", "may", "tao", "tui",
    "bro", "sis", "look", "see", "watch", "this", "here", "yo", "hey",
}

_MEANINGLESS_TOKENS = {
    "viral", "fyp", "fpy", "xuhuong", "xu hướng", "xh", "tem", "first",
    "top", "hóng", "hong", "chấm", "cham", "up", "fl", "follow", "ib",
    "inbox", "link", "share", "cmt", "comment", "ok", "oke", "ừ", "uh",
    "ờ", "à", "ha", "hả", "ủa", "tr", "ui", "real", "fr",
}

_SPAM_AD_PATTERNS = [
    re.compile(r"\bvào\s+shop\s+em\b", flags=re.IGNORECASE),
    re.compile(r"\bsắm\s+đồ\b.*\bshop\b", flags=re.IGNORECASE),
    re.compile(r"\binbox\s+(?:tư\s+vấn|mua|đặt|ngay)\b", flags=re.IGNORECASE),
    re.compile(r"\b(?:liên\s+hệ|lh|add|nhắn)\s*zalo\b", flags=re.IGNORECASE),
    re.compile(r"\btuyển\s+(?:ctv|cộng\s+tác\s+viên|sỉ|lẻ)\b", flags=re.IGNORECASE),
    re.compile(r"\bkiếm\s+tiền\s+online\b", flags=re.IGNORECASE),
    re.compile(r"\bgiỏ\s+hàng\s+góc\s+trái\b", flags=re.IGNORECASE),
    re.compile(r"\bsăn\s+sale\b", flags=re.IGNORECASE),
    re.compile(r"\blink\s+mua\b", flags=re.IGNORECASE),
    re.compile(r"(?<!\d)0[35789]\d{8}(?!\d)"),
]

_VI_NEUTRAL_STOPWORDS = {
    "là", "của", "và", "các", "những", "thì", "mà", "ở", "tại", "cho",
    "về", "với", "từ", "trong", "ngoài", "trên", "dưới", "khi", "lúc",
    "này", "đó", "kia", "ấy", "nọ", "một", "hai", "ba", "bốn", "năm",
    "cái", "con", "chiếc", "người", "ai", "gì", "nào", "đâu", "sao",
    "vậy", "thế", "nhé", "nha", "ạ", "ơi", "á", "à", "ừ", "ờ", "thôi",
    "lun", "luôn", "rồi", "òi", "đã", "đang", "sẽ", "cũng", "đều",
    "vẫn", "còn", "lại", "ra", "vào", "lên", "xuống", "đi", "đến", "tới",
    "bằng", "như", "theo", "để", "nên", "nếu", "vì", "do", "bởi", "tuy",
    "nhưng", "hoặc", "hay_là", "rằng", "mình", "tui", "tôi", "tao", "mày",
    "bạn", "anh", "chị", "em", "ông", "bà", "ổng", "bả", "ảnh", "chú",
    "cô", "bác", "mọi_người", "người_ta",
}

_EN_NEUTRAL_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when",
    "where", "why", "how", "what", "which", "who", "whom", "whose",
    "this", "that", "these", "those", "is", "am", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "must", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "up", "about", "into", "over", "after", "before", "under", "again",
    "further", "once", "here", "there", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "so", "than", "too", "very", "just", "now", "i", "me", "my",
    "myself", "we", "our", "ours", "you", "your", "yours", "he", "him",
    "his", "she", "her", "hers", "it", "its", "they", "them", "their",
    "im", "i'm", "it's", "that's", "there's", "don't", "doesn't", "didn't",
    "can't", "won't", "isn't", "aren't", "as", "because", "until", "while",
    "get", "got", "going", "go", "make", "made", "one", "two", "see",
    "say", "says", "said", "way", "bro", "man", "guy", "guys",
}

_EN_SIGNAL_WORDS = _EN_NEUTRAL_STOPWORDS | {
    "peak", "peakk", "peakkk", "fire", "cute", "love", "hate", "good",
    "bad", "nice", "wow", "edit", "edits", "movie", "movies", "video",
    "scared", "afraid", "holy", "shit", "fuck", "fuckin", "fucking",
    "freaking", "smooth", "damn", "best", "worst", "sad", "happy",
    "angry", "lol", "lmao", "omg", "wtf", "ngl", "tbh", "goat", "goated",
    "amazing", "incredible", "awesome", "cool", "insane", "crazy", "merry",
    "christmas", "time", "left", "repost", "reposts", "task", "mask",
    "think", "wants", "want", "family", "feel", "feeling", "quiet",
    "ride", "angel", "gold", "hard", "find", "even", "used", "less",
    "witnessing", "transition", "transitions", "cinema", "masterpiece",
    "cringe", "trash", "mid", "boring", "disappointed", "terrible",
    "awesome", "great", "perfect", "december", "doom", "marvel",
}

_VI_UNACCENTED_WORDS = {
    "khong", "khum", "hong", "duoc", "dc", "khong", "qua", "lam",
    "vui", "buon", "ghet", "thuong", "yeu", "thich", "nguoi", "minh",
    "tui", "may", "tao", "anh", "chi", "em", "ong", "ba", "troi",
    "choy", "nha", "nhe", "nhaa", "lun", "thui", "zay", "vay", "sao",
    "nao", "dau", "roi", "oi", "ne", "kia", "cai", "nay", "kia", "lam",
    "xem", "coi", "hay", "te", "do", "xau", "dep", "gioi", "dinh",
    "ghec", "chan", "oai", "kieu", "nhin", "nghe", "noi", "biet", "bt",
}

# Từ phổ biến của các ngôn ngữ Latin khác (Tây Ban Nha, Pháp, Bồ Đào Nha, Indo, Đức, Ý, Thổ)
_OTHER_LATIN_STOPWORDS = {
    # Spanish / Portuguese
    "que", "para", "muy", "esto", "pero", "gracias", "hola", "como", "una",
    "los", "las", "por", "con", "não", "muito", "você", "uma", "isso", "obrigado",
    "todo", "toda", "todos", "el", "la", "de", "en", "pro", "cada", "dia", "día",
    "buen", "buena", "bueno", "este", "esta", "está", "são", "mas", "dos", "nos",
    "tudo", "mais", "odio", "raiva", "nada", "destruir", "publicación", "emocionada",
    "abandonem", "amargura", "perdoem", "outros",
    # French
    "c'est", "très", "je", "suis", "merci", "avec", "pour", "pas", "une", "les", "mais",
    # Indonesian / Malay
    "yang", "tidak", "ini", "dengan", "untuk", "sekali", "aku", "kamu", "dan", "dari", "banget",
    # German / Italian / Turkish
    "ich", "und", "nicht", "das", "ist", "sehr", "ein", "mit", "die", "der",
    "che", "per", "sono", "molto", "grazie", "bir", "çok", "için", "değil",
}

_LAUGH_EMOJI_PATTERN = re.compile(r"[😂🤣😆🤡]|:\)\)+|=+\)+")
_NEGATIVE_SIGNAL_WORDS = (
    "ghét", "thù", "vô duyên", "mắc ói", "cay cú", "chán", "trù",
    "khinh", "đơm đặt", "tự ti", "ganh tị", "phốt", "drama", "tệ",
    "dở", "lố", "tào lao", "báo công an", "tội nghiệp", "cảm lạnh",
    "oải", "nặng nề", "hate", "trash", "mid", "awful", "terrible",
    "worst", "cringe", "boring", "disappointed",
)
_QUOTED_IRONY_PATTERN = re.compile(r'["“][^"”]{4,}["”]')

_OPINION_OR_EMOTION_MARKERS = {
    # Tiếng Việt
    "hay", "dở", "tệ", "ghét", "thích", "yêu", "thương", "vui", "buồn",
    "giận", "tức", "sợ", "lo", "đẹp", "xấu", "giỏi", "ngu", "khùng",
    "đỉnh", "chán", "tội", "khổ", "thật", "đúng", "sai", "không", "chẳng",
    "chưa", "quá", "lắm", "duyên", "tử tế", "vô duyên", "bình thường",
    "hóng", "ảo", "ganh",
    # Tiếng Anh
    "holy", "fuck", "fuckin", "fucking", "shit", "peak", "fire", "cute",
    "good", "bad", "nice", "wow", "no", "way", "love", "hate", "scared",
    "damn", "smooth", "crazy", "insane", "amazing", "incredible", "awesome",
    "best", "worst", "lol", "lmao", "omg", "wtf", "transition", "edit",
    "great", "cool", "mid", "trash", "cringe", "sad", "happy", "angry",
}


@dataclass
class PreprocessedComment:
    """Kết quả tiền xử lý, lọc nhiễu và nhận diện ngôn ngữ cho một bình luận."""

    original_text: str
    cleaned_text: str
    segmented_text: str
    ml_text: str
    tokens: list[str]
    language: str  # "vi" | "en" | "other"
    is_valid: bool
    filter_reason: str
    is_duplicate: bool
    has_sarcasm_signal: bool
    matched_video_terms: list[str] = field(default_factory=list)
    context_relevance: float = 0.0
    is_on_topic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_context_dict(media_context: Any) -> dict[str, Any] | None:
    if media_context is None:
        return None
    if hasattr(media_context, "to_dict"):
        data = media_context.to_dict()
        return data if isinstance(data, dict) else None
    if isinstance(media_context, dict):
        return media_context
    return None


def _split_context_term_candidates(raw_items: Sequence[Any]) -> list[str]:
    """Tách các thực thể/từ khóa từ trường danh sách (gồm cả cụm trong ngoặc đơn và dấu phẩy)."""
    combined_stopwords = _VI_NEUTRAL_STOPWORDS | _EN_NEUTRAL_STOPWORDS | _MEANINGLESS_TOKENS
    candidates: list[str] = []
    seen_low: set[str] = set()

    def _add(term: str) -> None:
        cleaned = _WHITESPACE_PATTERN.sub(" ", term.strip(" .,!?;:\"'()[]{}")).strip()
        if len(cleaned) < 2:
            return
        low = cleaned.lower()
        if low in combined_stopwords or low in seen_low:
            return
        if not _ALPHA_PATTERN.search(cleaned):
            return
        seen_low.add(low)
        candidates.append(cleaned)

    for raw in raw_items:
        text = str(raw or "").strip()
        if not text:
            continue
        # Tách phần ngoài ngoặc và phần trong ngoặc đơn: "Các cường quốc (Mỹ, Châu Âu)"
        for group in re.findall(r"\(([^)]+)\)", text):
            for part in re.split(r"[,;/|]", group):
                _add(part)
        outer = re.sub(r"\([^)]*\)", " ", text)
        for part in re.split(r"[,;/|]", outer):
            _add(part)
        _add(text)

    return candidates


def _extract_media_context_profile(media_context: Any) -> dict[str, Any] | None:
    """Trích xuất tập thuật ngữ bảo vệ và văn bản tham chiếu ngữ cảnh từ media_context."""
    data = _coerce_context_dict(media_context)
    if not data:
        return None

    raw_entities = data.get("doi_tuong_chinh") or []
    raw_keywords = data.get("tu_khoa") or []
    if isinstance(raw_entities, str):
        raw_entities = [s.strip() for s in raw_entities.split(",") if s.strip()]
    if isinstance(raw_keywords, str):
        raw_keywords = [s.strip() for s in raw_keywords.split(",") if s.strip()]

    terms = _split_context_term_candidates([*raw_entities, *raw_keywords])
    protected_terms = {t.lower() for t in terms}

    text_parts: list[str] = []
    for key in ("chu_de_chinh", "tom_tat_ngan", "quan_diem_noi_dung", "caption"):
        val = str(data.get(key) or "").strip()
        if val:
            text_parts.append(val)
    for key in ("doi_tuong_chinh", "tu_khoa", "chi_tiet_va_luan_diem", "chu_tren_man_hinh"):
        items = data.get(key) or []
        if isinstance(items, str):
            items = [items]
        for item in items:
            s = str(item or "").strip()
            if s:
                text_parts.append(s)
    transcript = str(data.get("transcript") or "").strip()
    if transcript:
        # Bỏ nhãn mốc thời gian [MM:SS-MM:SS] và giới hạn độ dài để không làm loãng TF-IDF
        clean_tr = re.sub(r"\[\d{2}:\d{2}-\d{2}:\d{2}\]", " ", transcript)
        text_parts.append(clean_tr[:1500])

    raw_ref = " ".join(text_parts)
    raw_ref = _URL_PATTERN.sub(" ", raw_ref)
    raw_ref = raw_ref.replace("#", " ")
    reference_doc = _build_context_bow_doc(raw_ref)

    caption_part = str(data.get("caption") or "").strip()
    clean_tr = re.sub(r"\[\d{2}:\d{2}-\d{2}:\d{2}\]", " ", transcript)[:1500] if transcript else ""
    raw_media_text = _URL_PATTERN.sub(" ", f"{caption_part} {clean_tr}").replace("#", " ")
    media_text_doc = _build_context_bow_doc(raw_media_text)

    if not terms and not reference_doc and not media_text_doc:
        return None

    return {
        "terms": terms,
        "protected_terms": protected_terms,
        "reference_doc": reference_doc,
        "media_text_doc": media_text_doc,
    }


def _build_context_bow_doc(text: str) -> str:
    """Chuẩn hóa chuỗi giữ nguyên khoảng trắng từ ghép để so khớp TF-IDF giữa comment và video."""
    combined_stopwords = _VI_NEUTRAL_STOPWORDS | _EN_NEUTRAL_STOPWORDS
    cleaned = clean_comment_text(text, normalize_teencode=False)
    tokens = re.findall(r"[^\W_][\w']*", cleaned.lower(), flags=re.UNICODE)
    kept = [
        tok.strip("'")
        for tok in tokens
        if len(tok.strip("'")) >= 2
        and tok.strip("'") not in combined_stopwords
        and _ALPHA_PATTERN.search(tok)
    ]
    return " ".join(kept)


def _matches_any_term(text: str, terms: Sequence[str] | set[str]) -> list[str]:
    """Trả về danh sách các thuật ngữ/thực thể video xuất hiện nguyên vẹn trong câu."""
    if not text or not terms:
        return []
    matched: list[str] = []
    seen_low: set[str] = set()
    for term in terms:
        t = str(term).strip()
        if not t:
            continue
        low = t.lower()
        if low in seen_low:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(t)}(?!\w)", flags=re.IGNORECASE | re.UNICODE)
        if pattern.search(text):
            seen_low.add(low)
            matched.append(t)
    return matched


def _score_context_relevance(
    valid_docs: Sequence[str],
    reference_doc: str,
    matched_lists: Sequence[list[str]],
    media_text_doc: str = "",
) -> list[tuple[float, bool]]:
    """Tính TF-IDF cosine similarity + khớp từ khóa để đo độ bám sát nội dung video."""
    if not valid_docs:
        return []

    ref_docs = [d for d in (reference_doc.strip(), media_text_doc.strip()) if d]
    sims = np.zeros(len(valid_docs), dtype=float)
    if ref_docs:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vec = TfidfVectorizer(
                token_pattern=r"(?u)\b[\w']{2,}\b",
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
            )
            mat = vec.fit_transform([*ref_docs, *valid_docs])
            n_refs = len(ref_docs)
            ref_mat = mat[:n_refs]
            doc_mat = mat[n_refs:]
            sim_mat = np.asarray((doc_mat @ ref_mat.T).toarray())
            sims = sim_mat.max(axis=1)
        except ValueError:
            sims = np.zeros(len(valid_docs), dtype=float)

    results: list[tuple[float, bool]] = []
    for idx, matched in enumerate(matched_lists):
        cos_sim = float(sims[idx]) if idx < len(sims) else 0.0
        term_bonus = min(0.60, 0.25 * len(matched))
        relevance = round(min(1.0, max(0.0, cos_sim * 1.8 + term_bonus)), 4)
        is_on_topic = bool(matched) or (relevance >= 0.12)
        results.append((relevance, is_on_topic))
    return results


def normalize_tiktok_emojis(text: str) -> str:
    """Chuyển đổi sticker/emoji dạng chữ TikTok ([Cười lớn]...) sang emoji Unicode."""
    return _TIKTOK_EMOJI_PATTERN.sub(
        lambda m: f" {TIKTOK_TEXT_EMOJI_MAP.get(m.group(0).lower(), '')} ",
        text,
    )


def detect_language(text: str) -> str:
    """Nhận diện ngôn ngữ bình luận: 'vi' (Tiếng Việt), 'en' (Tiếng Anh), hoặc 'other'."""
    sample = _MENTION_PATTERN.sub(" ", text or "")
    sample = _URL_PATTERN.sub(" ", sample)
    sample = _HASHTAG_PATTERN.sub(" ", sample).strip()
    if not sample:
        return "vi"

    alpha_chars = [ch for ch in sample if ch.isalpha()]
    if not alpha_chars:
        return "vi"

    # 1. Phát hiện chữ viết phi-Latin (Trung, Nhật, Hàn, Nga, Ả Rập, Thái...)
    non_latin_cnt = len(_NON_LATIN_SCRIPT_PATTERN.findall(sample))
    if non_latin_cnt / max(1, len(alpha_chars)) >= 0.35:
        return "other"

    # 2. Ký tự thuần Việt đặc trưng (ă, đ, ơ, ư, thanh hỏi/nặng, dấu kép...) -> Tiếng Việt
    if _VI_UNIQUE_DIACRITIC_PATTERN.search(sample):
        return "vi"

    words = [
        w.strip(".,!?;:\"'()[]{}").lower()
        for w in sample.split()
        if w.strip(".,!?;:\"'()[]{}")
    ]
    if not words:
        return "vi"

    vi_hits = sum(1 for w in words if w in _VI_UNACCENTED_WORDS)
    en_hits = sum(1 for w in words if w in _EN_SIGNAL_WORDS)
    other_latin_hits = sum(1 for w in words if w in _OTHER_LATIN_STOPWORDS)

    # 3. Nếu có ký tự riêng ngôn ngữ Latin khác (ñ, ¿, ¡, ç...) hoặc >= 2 từ ngoại ngữ Latin -> 'other'
    if _OTHER_LATIN_UNIQUE_CHARS.search(sample) or (
        other_latin_hits >= 2 and other_latin_hits > en_hits and vi_hits == 0
    ):
        return "other"

    # 4. Dấu Latin chung (á, é, ó, ê, ô, à...) khi không khớp ngoại ngữ Latin -> Tiếng Việt
    if _VI_DIACRITIC_PATTERN.search(sample):
        return "vi"

    if vi_hits >= 2 and vi_hits >= en_hits:
        return "vi"
    if en_hits >= 1 and en_hits >= vi_hits:
        return "en"
    if vi_hits == 1 and en_hits == 0:
        return "vi"

    # Chuỗi chỉ dùng ASCII chuẩn mà không khớp từ khóa Việt -> ưu tiên tiếng Anh
    if all(ord(ch) < 128 for ch in "".join(alpha_chars)):
        return "en"

    return "vi"


def clean_comment_text(
    text: str,
    normalize_teencode: bool = True,
    language: str | None = None,
) -> str:
    """Làm sạch bình luận giữ nguyên emoji; chỉ áp dụng teencode tiếng Việt khi là tiếng Việt."""
    if text is None:
        return ""
    cleaned = str(text).strip()
    if not cleaned:
        return ""

    cleaned = normalize_tiktok_emojis(cleaned)
    cleaned = _URL_PATTERN.sub(" ", cleaned)
    cleaned = _REPEAT_CHAR_PATTERN.sub(r"\1\1", cleaned)

    lang = language or detect_language(cleaned)
    if normalize_teencode and lang == "vi":
        cleaned = _TEENCODE_PATTERN.sub(
            lambda m: TEENCODE_MAP.get(m.group(1).lower(), m.group(1)),
            cleaned,
        )
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def _safe_word_tokenize(text: str, language: str = "vi") -> list[str]:
    """Tách từ tiếng Việt bằng underthesea, hoặc tách từ tiếng Anh chuẩn."""
    if not text.strip():
        return []
    if language == "en":
        return [
            tok
            for tok in re.findall(r"[A-Za-z0-9']+|[^\w\s]", text, flags=re.UNICODE)
            if tok.strip()
        ]
    try:
        from underthesea import word_tokenize

        tokenized = word_tokenize(text, format="text")
        return [tok for tok in tokenized.split() if tok.strip()]
    except Exception:
        return [tok for tok in re.findall(r"[\w_]+|[^\w\s]", text, flags=re.UNICODE) if tok.strip()]


def _safe_pos_tag(text: str) -> list[tuple[str, str]]:
    """Gán nhãn từ loại tiếng Việt bằng underthesea."""
    if not text.strip():
        return []
    try:
        from underthesea import pos_tag

        return pos_tag(text)
    except Exception:
        return [(tok, "X") for tok in text.split()]


def _safe_ner(text: str) -> list[tuple[str, str, str, str]]:
    """Nhận dạng thực thể tên riêng bằng underthesea."""
    if not text.strip():
        return []
    try:
        from underthesea import ner

        return ner(text)
    except Exception:
        return []


def _strip_fancy_unicode(text: str) -> str:
    """Chuẩn hóa ký tự toán học/cách điệu (𝓐𝓻𝓲𝓮𝓵𝓵𝓪 -> Ariella) để kiểm tra từ đơn giản."""
    return unicodedata.normalize("NFKC", text)


def _is_name_only_comment(
    cleaned: str,
    language: str,
    protected_terms: set[str] | None = None,
) -> bool:
    """Phát hiện bình luận ngắn chỉ là tên bạn bè (không có @) dùng underthesea NER/POS."""
    # Không lọc nhầm thực thể/đối tượng trọng tâm của video
    if protected_terms and _matches_any_term(cleaned, protected_terms):
        return False

    # Không bao giờ lọc nhầm bình luận viết IN HOA toàn bộ ("HOLY TRANSITION", "NO FUCKIN WAY")
    alpha_chars = [c for c in cleaned if c.isalpha()]
    if alpha_chars and all(c.isupper() for c in alpha_chars):
        return False

    words = [w.strip(".,!?;:\"'()[]{}") for w in cleaned.split()]
    words = [w for w in words if w]
    if not words or len(words) > 4:
        return False

    lower_words = {_strip_fancy_unicode(w).lower() for w in words}
    if lower_words & _OPINION_OR_EMOTION_MARKERS:
        return False

    remaining = [w for w in words if w.lower() not in _FRIEND_CALL_WORDS]
    if not remaining:
        return True
    if len(remaining) < 2:
        return False

    # Với câu tiếng Anh ngắn không có @, chỉ lọc nếu toàn bộ là Title-Case 2 từ kiểu tên riêng
    if language == "en":
        return False

    try:
        ner_items = _safe_ner(" ".join(remaining))
        if ner_items:
            all_per_or_np = True
            has_per_or_np = False
            for item in ner_items:
                tok = str(item[0]).strip()
                pos = str(item[1]) if len(item) > 1 else ""
                ner_tag = str(item[3]) if len(item) > 3 else "O"
                if ner_tag.endswith("PER") or pos == "Np":
                    has_per_or_np = True
                elif tok.lower() in _FRIEND_CALL_WORDS or not _ALPHA_PATTERN.search(tok):
                    continue
                else:
                    all_per_or_np = False
                    break
            if has_per_or_np and all_per_or_np:
                if any(tok[:1].isupper() for tok in remaining):
                    return True
    except Exception:
        pass

    return False


def _detect_filter_reason(
    raw_text: str,
    cleaned_text: str,
    language: str,
    protected_terms: set[str] | None = None,
) -> str:
    """Kiểm tra bình luận nhiễu / vô nghĩa / ngôn ngữ không hỗ trợ và trả về mã lý do."""
    raw_stripped = (raw_text or "").strip()
    if not raw_stripped:
        return "FILTERED_EMPTY"

    no_url = _URL_PATTERN.sub(" ", raw_stripped).strip()
    no_hashtag = _HASHTAG_PATTERN.sub(" ", no_url).strip()
    if not no_hashtag or _PURE_PUNCT_OR_EMOJI_PATTERN.match(no_hashtag):
        return "FILTERED_URL_OR_SYMBOL"

    if not cleaned_text or not _ALPHA_PATTERN.search(cleaned_text):
        return "FILTERED_URL_OR_SYMBOL"

    # Kiểm tra tag bạn bè có @mention (kể cả @jacr$ryan-J, @Trường Giang, @Psychic Astrology)
    if "@" in raw_stripped:
        without_mentions = _MENTION_PATTERN.sub(" ", cleaned_text)
        without_mentions_words = [
            _strip_fancy_unicode(w).strip(".,!?;:\"'()[]{}")
            for w in without_mentions.split()
        ]
        rem_words = [
            w
            for w in without_mentions_words
            if w and _ALPHA_PATTERN.search(w) and w.lower() not in _FRIEND_CALL_WORDS
        ]
        if not rem_words:
            return "FILTERED_TAG_FRIEND"
        # Nếu bắt đầu bằng @ và phần còn lại chỉ là 1-2 từ tên hiển thị ngắn không chứa từ ý kiến
        if raw_stripped.startswith("@") and len(rem_words) <= 2:
            if not ({w.lower() for w in rem_words} & _OPINION_OR_EMOTION_MARKERS):
                return "FILTERED_TAG_FRIEND"

    # Kiểm tra quảng cáo / spam bán hàng
    for ad_pat in _SPAM_AD_PATTERNS:
        if ad_pat.search(raw_stripped) or ad_pat.search(cleaned_text):
            return "FILTERED_SPAM_AD"

    # Lọc ngôn ngữ không thuộc Tiếng Việt hoặc Tiếng Anh
    if language == "other":
        return "FILTERED_UNSUPPORTED_LANG"

    matches_video_entity = bool(protected_terms and _matches_any_term(cleaned_text, protected_terms))

    # Kiểm tra các từ vô nghĩa / seeding đơn lẻ (không lọc nếu nhắc đúng thực thể video)
    alpha_only = re.sub(r"[^\w\s]", " ", cleaned_text, flags=re.UNICODE)
    alpha_only = _WHITESPACE_PATTERN.sub(" ", alpha_only).strip().lower()
    if not alpha_only:
        return "FILTERED_URL_OR_SYMBOL"
    if not matches_video_entity:
        if alpha_only in _MEANINGLESS_TOKENS:
            return "FILTERED_MEANINGLESS"
        words = alpha_only.split()
        if len(words) <= 2 and all(w in _MEANINGLESS_TOKENS for w in words):
            return "FILTERED_MEANINGLESS"

    # Kiểm tra bình luận chỉ gọi tên bạn bè không có @
    if _is_name_only_comment(cleaned_text, language=language, protected_terms=protected_terms):
        return "FILTERED_NAME_ONLY"

    return ""


def _build_ml_text(tokens: list[str], language: str) -> str:
    """Tạo chuỗi văn bản chuẩn cho ML cổ điển (giữ từ phủ định & cường độ, bỏ stopword trung tính)."""
    stopwords = _EN_NEUTRAL_STOPWORDS if language == "en" else _VI_NEUTRAL_STOPWORDS
    kept: list[str] = []
    for tok in tokens:
        low = tok.lower().strip()
        if not low or not _ALPHA_PATTERN.search(low):
            continue
        if low in stopwords:
            continue
        kept.append(low)
    return " ".join(kept)


def _has_sarcasm_signal(original_text: str, cleaned_text: str) -> bool:
    """Phát hiện dấu hiệu mỉa mai/châm biếm để ưu tiên theo dõi."""
    low = cleaned_text.lower()
    has_laugh = bool(
        _LAUGH_EMOJI_PATTERN.search(cleaned_text)
        or _LAUGH_EMOJI_PATTERN.search(original_text)
    )
    has_neg_word = any(w in low for w in _NEGATIVE_SIGNAL_WORDS)
    has_quote = bool(_QUOTED_IRONY_PATTERN.search(original_text))
    return (has_laugh and has_neg_word) or (has_quote and has_laugh)


class TextPreprocessor:
    """Giai đoạn 2: Lọc bình luận rác, nhận diện ngôn ngữ (VI/EN) & chuẩn hóa với underthesea."""

    def __init__(
        self,
        normalize_teencode: bool = True,
        enable_noise_filter: bool = True,
    ) -> None:
        self.normalize_teencode = normalize_teencode
        self.enable_noise_filter = enable_noise_filter

    def process_one(
        self,
        text: str,
        seen_normalized: set[str] | None = None,
        media_context: Any = None,
        _profile: dict[str, Any] | None = None,
    ) -> PreprocessedComment:
        profile = _profile if _profile is not None else _extract_media_context_profile(media_context)
        protected_terms = profile["protected_terms"] if profile else None

        original = "" if text is None else str(text)
        lang = detect_language(original)
        cleaned = clean_comment_text(
            original,
            normalize_teencode=self.normalize_teencode,
            language=lang,
        )

        filter_reason = ""
        if self.enable_noise_filter:
            filter_reason = _detect_filter_reason(
                original,
                cleaned,
                language=lang,
                protected_terms=protected_terms,
            )
        elif not cleaned:
            filter_reason = "FILTERED_EMPTY"

        is_valid = filter_reason == ""

        analysis_clean = _WHITESPACE_PATTERN.sub(
            " ", _MENTION_PATTERN.sub(" ", cleaned)
        ).strip()

        norm_key = _WHITESPACE_PATTERN.sub(" ", analysis_clean.lower()).strip()
        is_duplicate = False
        if is_valid and seen_normalized is not None and norm_key:
            if norm_key in seen_normalized:
                is_duplicate = True
            else:
                seen_normalized.add(norm_key)

        tokens = _safe_word_tokenize(analysis_clean, language=lang) if is_valid else []
        segmented_text = " ".join(tokens)
        ml_text = _build_ml_text(tokens, language=lang) if tokens else ""
        sarcasm_signal = _has_sarcasm_signal(original, cleaned) if is_valid else False

        matched_terms: list[str] = []
        relevance = 0.0
        on_topic = False
        if is_valid and profile is not None:
            matched_terms = _matches_any_term(analysis_clean, profile["terms"])
            if _profile is None:
                doc = _build_context_bow_doc(analysis_clean)
                scored = _score_context_relevance(
                    [doc],
                    profile["reference_doc"],
                    [matched_terms],
                    media_text_doc=profile.get("media_text_doc", ""),
                )
                if scored:
                    relevance, on_topic = scored[0]

        return PreprocessedComment(
            original_text=original,
            cleaned_text=analysis_clean if is_valid else cleaned,
            segmented_text=segmented_text,
            ml_text=ml_text,
            tokens=tokens,
            language=lang,
            is_valid=is_valid,
            filter_reason=filter_reason,
            is_duplicate=is_duplicate,
            has_sarcasm_signal=sarcasm_signal,
            matched_video_terms=matched_terms,
            context_relevance=relevance,
            is_on_topic=on_topic,
        )

    def process_batch(
        self,
        texts: Sequence[str],
        media_context: Any = None,
    ) -> list[PreprocessedComment]:
        seen_normalized: set[str] = set()
        profile = _extract_media_context_profile(media_context)
        results = [
            self.process_one(
                t,
                seen_normalized=seen_normalized,
                media_context=media_context,
                _profile=profile,
            )
            for t in texts
        ]
        if profile is not None:
            valid_indices = [i for i, item in enumerate(results) if item.is_valid]
            if valid_indices:
                valid_docs = [
                    _build_context_bow_doc(results[i].cleaned_text) for i in valid_indices
                ]
                matched_lists = [results[i].matched_video_terms for i in valid_indices]
                scored = _score_context_relevance(
                    valid_docs,
                    profile["reference_doc"],
                    matched_lists,
                    media_text_doc=profile.get("media_text_doc", ""),
                )
                for idx, (rel, on_topic) in zip(valid_indices, scored):
                    results[idx].context_relevance = rel
                    results[idx].is_on_topic = on_topic
        return results


def extract_keywords_tfidf(
    comments: Sequence[PreprocessedComment],
    polarity_labels: Sequence[str] | None = None,
    top_k: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Trích xuất từ khóa nổi bật (song ngữ VI + EN, loại bỏ đầy đủ stopword cả 2 ngôn ngữ)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    combined_stopwords = _VI_NEUTRAL_STOPWORDS | _EN_NEUTRAL_STOPWORDS
    valid_pairs: list[tuple[str, str]] = []

    for idx, item in enumerate(comments):
        if not item.is_valid:
            continue
        lbl = polarity_labels[idx] if polarity_labels and idx < len(polarity_labels) else "ALL"

        if item.language == "en":
            filtered_words = [
                tok.lower().strip("'")
                for tok in item.tokens
                if _ALPHA_PATTERN.search(tok)
                and len(tok.strip("'")) >= 2
                and tok.lower() not in combined_stopwords
                and tok.lower().strip("'") not in combined_stopwords
            ]
        else:
            pos_items = _safe_pos_tag(item.cleaned_text)
            filtered_words = []
            for word, tag in pos_items:
                norm_w = word.strip().lower().replace(" ", "_")
                if not norm_w or not _ALPHA_PATTERN.search(norm_w):
                    continue
                if norm_w in combined_stopwords or len(norm_w) < 2:
                    continue
                if tag.startswith(("N", "V", "A")) or tag == "X":
                    filtered_words.append(norm_w)

        doc = " ".join(filtered_words) if filtered_words else item.ml_text
        if doc.strip():
            valid_pairs.append((doc, lbl))

    def _top_terms(docs: list[str]) -> list[dict[str, Any]]:
        if not docs:
            return []
        try:
            vec = TfidfVectorizer(
                token_pattern=r"(?u)\b[\w_]{2,}\b",
                ngram_range=(1, 2),
                max_features=300,
                min_df=1,
            )
            mat = vec.fit_transform(docs)
            scores = np.asarray(mat.mean(axis=0)).ravel()
            names = vec.get_feature_names_out()
            order = np.argsort(scores)[::-1][:top_k]
            return [
                {
                    "keyword": str(names[i]).replace("_", " "),
                    "tfidf_score": round(float(scores[i]), 4),
                }
                for i in order
                if scores[i] > 0
            ]
        except ValueError:
            return []

    all_docs = [d for d, _ in valid_pairs]
    pos_docs = [d for d, lbl in valid_pairs if lbl == "POS"]
    neg_docs = [d for d, lbl in valid_pairs if lbl == "NEG"]

    return {
        "overall": _top_terms(all_docs),
        "positive": _top_terms(pos_docs),
        "negative": _top_terms(neg_docs),
    }
