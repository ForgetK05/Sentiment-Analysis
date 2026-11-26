from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from src.ingestion.tiktok_collector import (
    TikTokMediaBundle,
    collect_tiktok_post,
    extract_post_id,
    load_comments_dataframe,
)
from src.models.gemma_auditor import GemmaAuditItem, GemmaAuditor, build_model_comparison
from src.models.media_context import IMAGE_EXTS, VIDEO_EXTS, MediaAnalysisResult, MediaAnalyzer
from src.models.sentiment_engine import (
    DEFAULT_EN_EMOTION_MODEL,
    DEFAULT_EN_SENTIMENT_MODEL,
    DEFAULT_PHOBERT_EMOTION_MODEL,
    DEFAULT_PHOBERT_SENTIMENT_MODEL,
    DEFAULT_VISOBERT_MODEL,
    LABEL_VI_MAP,
    AnalyzedComment,
    CommentAnalyzer,
)
from src.preprocessing.text_cleaner import (
    PreprocessedComment,
    TextPreprocessor,
    extract_keywords_tfidf,
)
from src.visualization import plot_sentiment_dashboard


MEDIA_CONTEXT_FILENAME = "media_context.json"

_LANG_VI_MAP: dict[str, str] = {
    "vi": "Tiếng Việt",
    "en": "Tiếng Anh",
    "other": "Ngôn ngữ khác",
}

FILTER_REASON_VI_MAP: dict[str, str] = {
    "FILTERED_EMPTY": "Bình luận rỗng",
    "FILTERED_UNSUPPORTED_LANG": "Ngôn ngữ khác ngoài Tiếng Việt / Tiếng Anh",
    "FILTERED_URL_OR_SYMBOL": "Chỉ chứa URL, số hoặc ký tự/emoji",
    "FILTERED_TAG_FRIEND": "Chỉ gắn thẻ (@tag) bạn bè",
    "FILTERED_NAME_ONLY": "Chỉ gọi tên riêng (không có cảm xúc)",
    "FILTERED_SPAM_AD": "Quảng cáo / spam bán hàng",
    "FILTERED_MEANINGLESS": "Từ rác / seeding ngắn vô nghĩa",
}

_FINAL_CORE_COLUMNS: tuple[str, ...] = (
    "stt",
    "original_text",
    "polarity_vi",
    "polarity_confidence",
    "emotion_vi",
    "emotion_confidence",
    "likes",
    "comment_type",
    "username",
    "language",
    "created_at",
    "reply_count",
)

_FINAL_GEMMA_COLUMNS: tuple[str, ...] = (
    "gemma_polarity_vi",
    "gemma_emotion_vi",
    "gemma_is_sarcasm",
    "gemma_reason",
)

FINAL_COLUMN_HEADERS: dict[str, str] = {
    "stt": "stt\n(số thứ tự)",
    "original_text": "original_text\n(nội dung bình luận)",
    "polarity_vi": "polarity_vi\n(sắc thái)",
    "polarity_confidence": "polarity_confidence\n(độ tự tin sắc thái %)",
    "emotion_vi": "emotion_vi\n(cảm xúc)",
    "emotion_confidence": "emotion_confidence\n(độ tự tin cảm xúc %)",
    "gemma_polarity_vi": "gemma_polarity_vi\n(sắc thái gemma)",
    "gemma_emotion_vi": "gemma_emotion_vi\n(cảm xúc gemma)",
    "gemma_is_sarcasm": "gemma_is_sarcasm\n(nói mỉa theo gemma)",
    "gemma_reason": "gemma_reason\n(lý do gemma giải thích)",
    "likes": "likes\n(lượt thích)",
    "comment_type": "comment_type\n(loại bình luận)",
    "username": "username\n(tài khoản)",
    "language": "language\n(ngôn ngữ)",
    "created_at": "created_at\n(thời gian đăng)",
    "reply_count": "reply_count\n(số phản hồi)",
}

_DETAILED_CORE_COLUMNS: tuple[str, ...] = (
    "stt",
    "comment_id",
    "parent_comment_id",
    "username",
    "nickname",
    "likes",
    "language",
    "original_text",
    "cleaned_text",
    "segmented_text",
    "is_duplicate",
    "has_sarcasm_signal",
    "needs_audit",
    "is_on_topic",
    "context_relevance",
    "matched_video_terms",
    "context_alignment",
    "polarity_label",
    "polarity_score",
    "score_POS",
    "score_NEG",
    "score_NEU",
    "emotion_label",
    "pred_nb",
    "pred_cnb",
    "pred_lr",
    "pred_ridge",
    "pred_sgd",
    "pred_svm",
    "pred_nbsvm",
    "pred_rbf_svm",
    "pred_rf",
    "pred_tree",
    "pred_ml_ensemble",
    "pred_lexicon",
    "dl_primary_model",
    "pred_phobert",
    "pred_visobert",
    "pred_roberta",
    "emotion_ml_label",
    "pred_emo_cnb",
    "pred_emo_lr",
    "pred_emo_ridge",
    "pred_emo_svm",
    "pred_emo_nbsvm",
    "pred_emo_sgd",
)

_DETAILED_GEMMA_COLUMNS: tuple[str, ...] = (
    "gemma_audited",
    "gemma_polarity",
    "gemma_emotion",
    "gemma_agrees_polarity",
    "gemma_agrees_emotion",
)

DETAILED_COLUMN_HEADERS: dict[str, str] = {
    "stt": "stt\n(số thứ tự)",
    "comment_id": "comment_id\n(mã bình luận)",
    "parent_comment_id": "parent_comment_id\n(mã bình luận gốc)",
    "username": "username\n(tài khoản)",
    "nickname": "nickname\n(tên hiển thị)",
    "likes": "likes\n(lượt thích)",
    "language": "language\n(ngôn ngữ)",
    "original_text": "original_text\n(nội dung gốc)",
    "cleaned_text": "cleaned_text\n(nội dung đã làm sạch)",
    "segmented_text": "segmented_text\n(văn bản đã tách từ)",
    "is_duplicate": "is_duplicate\n(bình luận trùng lặp)",
    "has_sarcasm_signal": "has_sarcasm_signal\n(dấu hiệu mỉa mai)",
    "needs_audit": "needs_audit\n(cần thẩm định kỹ)",
    "is_on_topic": "is_on_topic\n(bám sát chủ đề video)",
    "context_relevance": "context_relevance\n(độ liên quan video 0-1)",
    "matched_video_terms": "matched_video_terms\n(từ khóa video khớp)",
    "context_alignment": "context_alignment\n(liên hệ với nội dung video)",
    "polarity_label": "polarity_label\n(nhãn sắc thái chuẩn)",
    "polarity_score": "polarity_score\n(điểm cực tính -1 đến +1)",
    "score_POS": "score_POS\n(xác suất tích cực)",
    "score_NEG": "score_NEG\n(xác suất tiêu cực)",
    "score_NEU": "score_NEU\n(xác suất trung tính)",
    "emotion_label": "emotion_label\n(nhãn cảm xúc chuẩn)",
    "pred_nb": "pred_nb\n(dự đoán multinomial nb)",
    "pred_cnb": "pred_cnb\n(dự đoán complement nb)",
    "pred_lr": "pred_lr\n(dự đoán logistic regression)",
    "pred_ridge": "pred_ridge\n(dự đoán ridge classifier)",
    "pred_sgd": "pred_sgd\n(dự đoán sgd classifier)",
    "pred_svm": "pred_svm\n(dự đoán linear svm)",
    "pred_nbsvm": "pred_nbsvm\n(dự đoán nbsvm)",
    "pred_rbf_svm": "pred_rbf_svm\n(dự đoán rbf kernel svm)",
    "pred_rf": "pred_rf\n(dự đoán random forest)",
    "pred_tree": "pred_tree\n(dự đoán histgradientboosting)",
    "pred_ml_ensemble": "pred_ml_ensemble\n(dự đoán ml ensemble)",
    "pred_lexicon": "pred_lexicon\n(dự đoán từ điển luật)",
    "dl_primary_model": "dl_primary_model\n(mô hình học sâu sử dụng)",
    "pred_phobert": "pred_phobert\n(dự đoán phobert - vi)",
    "pred_visobert": "pred_visobert\n(dự đoán visobert - vi)",
    "pred_roberta": "pred_roberta\n(dự đoán twitter-roberta - en)",
    "emotion_ml_label": "emotion_ml_label\n(cảm xúc ml ensemble)",
    "pred_emo_cnb": "pred_emo_cnb\n(cảm xúc complement nb)",
    "pred_emo_lr": "pred_emo_lr\n(cảm xúc logistic regression)",
    "pred_emo_ridge": "pred_emo_ridge\n(cảm xúc ridge classifier)",
    "pred_emo_svm": "pred_emo_svm\n(cảm xúc linear svm)",
    "pred_emo_nbsvm": "pred_emo_nbsvm\n(cảm xúc nbsvm)",
    "pred_emo_sgd": "pred_emo_sgd\n(cảm xúc sgd classifier)",
    "gemma_audited": "gemma_audited\n(đã thẩm định gemma)",
    "gemma_polarity": "gemma_polarity\n(nhãn sắc thái gemma)",
    "gemma_emotion": "gemma_emotion\n(nhãn cảm xúc gemma)",
    "gemma_agrees_polarity": "gemma_agrees_polarity\n(khớp sắc thái với gemma)",
    "gemma_agrees_emotion": "gemma_agrees_emotion\n(khớp cảm xúc với gemma)",
}

_NOISE_COLUMNS: tuple[str, ...] = (
    "stt",
    "original_text",
    "filter_reason_vi",
    "filter_reason",
    "language",
    "likes",
    "username",
    "nickname",
    "comment_id",
)

NOISE_COLUMN_HEADERS: dict[str, str] = {
    "stt": "stt\n(số thứ tự)",
    "original_text": "original_text\n(nội dung gốc)",
    "filter_reason_vi": "filter_reason_vi\n(lý do lọc)",
    "filter_reason": "filter_reason\n(mã lý do lọc)",
    "language": "language\n(ngôn ngữ)",
    "likes": "likes\n(lượt thích)",
    "username": "username\n(tài khoản)",
    "nickname": "nickname\n(tên hiển thị)",
    "comment_id": "comment_id\n(mã bình luận)",
}

COMPARISON_COLUMN_HEADERS: dict[str, str] = {
    "task": "task\n(nhiệm vụ)",
    "model": "model\n(mô hình)",
    "family": "family\n(nhóm phương pháp)",
    "total_comments": "total_comments\n(số bình luận)",
    "pos_count": "pos_count\n(tích cực SL)",
    "neu_count": "neu_count\n(trung tính SL)",
    "neg_count": "neg_count\n(tiêu cực SL)",
    "pos_pct": "pos_pct\n(tích cực %)",
    "neu_pct": "neu_pct\n(trung tính %)",
    "neg_pct": "neg_pct\n(tiêu cực %)",
    "agreement_vs_gemma_pct": "agreement_vs_gemma_pct\n(khớp gemma %)",
    "macro_f1_vs_gemma": "macro_f1_vs_gemma\n(macro-f1 so với gemma)",
    "cohen_kappa_vs_gemma": "cohen_kappa_vs_gemma\n(cohen kappa so với gemma)",
    "inference_time_sec": "inference_time_sec\n(thời gian chạy giây)",
}


def _strip_row_keys(row: dict[str, Any]) -> dict[str, Any]:
    """Chuẩn hóa key về tên cột gốc và dọn giá trị NaN."""
    cleaned: dict[str, Any] = {}
    for k, v in row.items():
        base_k = str(k).split("\n", 1)[0].strip()
        if pd.isna(v):
            cleaned[base_k] = ""
        else:
            cleaned[base_k] = v
    return cleaned


def _has_gemma_audit_rows(norm_rows: Sequence[dict[str, Any]]) -> bool:
    return any(str(r.get("gemma_audited", "")).strip().lower() == "true" for r in norm_rows)


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        if val == "" or val is None or pd.isna(val):
            return default
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _format_pct_str(val: Any) -> str:
    try:
        if val == "" or val is None or pd.isna(val):
            return ""
        return f"{float(val) * 100.0:.1f}%"
    except (TypeError, ValueError):
        return ""


def _parse_bool_flag(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in {"true", "1", "yes", "y", "có"}


def _sort_rows_by_likes_desc(norm_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sắp xếp ổn định theo likes giảm dần để final_results.csv và detailed_model_outputs.csv khớp 1-1 theo STT."""
    return sorted(
        norm_rows,
        key=lambda r: _safe_int(r.get("likes"), 0),
        reverse=True,
    )


def format_final_results_df(clean_rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Tạo DataFrame `final_results.csv` (tiêu đề song ngữ 2 dòng Anh–Việt, 12 cột lõi hoặc 16 cột khi bật AI Audit, sắp theo lượt thích giảm dần)."""
    norm_rows = _sort_rows_by_likes_desc([_strip_row_keys(r) for r in clean_rows])
    has_gemma = _has_gemma_audit_rows(norm_rows)

    cols = list(_FINAL_CORE_COLUMNS[:6])
    if has_gemma:
        cols.extend(_FINAL_GEMMA_COLUMNS)
    cols.extend(_FINAL_CORE_COLUMNS[6:])

    out_rows: list[dict[str, Any]] = []
    for stt, r in enumerate(norm_rows, start=1):
        g_pol = str(r.get("gemma_polarity", "")).strip()
        g_audited = _parse_bool_flag(r.get("gemma_audited", False))
        lang_code = str(r.get("language", "")).strip().lower()
        row_out: dict[str, Any] = {
            "stt": stt,
            "original_text": r.get("original_text", ""),
            "polarity_vi": r.get("polarity_vi", ""),
            "polarity_confidence": _format_pct_str(r.get("polarity_confidence", "")),
            "emotion_vi": r.get("emotion_vi", ""),
            "emotion_confidence": _format_pct_str(r.get("emotion_confidence", "")),
            "gemma_polarity_vi": LABEL_VI_MAP.get(g_pol, g_pol) if g_audited else "",
            "gemma_emotion_vi": r.get("gemma_emotion_vi", "") if g_audited else "",
            "gemma_is_sarcasm": (
                ("Có" if _parse_bool_flag(r.get("gemma_is_sarcasm", False)) else "Không")
                if g_audited
                else ""
            ),
            "gemma_reason": r.get("gemma_reason", "") if g_audited else "",
            "likes": _safe_int(r.get("likes"), 0),
            "comment_type": "Trả lời" if _parse_bool_flag(r.get("is_reply", False)) else "Bình luận chính",
            "username": r.get("username", ""),
            "language": _LANG_VI_MAP.get(lang_code, lang_code),
            "created_at": r.get("created_at", ""),
            "reply_count": _safe_int(r.get("reply_count"), 0),
        }
        out_rows.append({c: row_out[c] for c in cols})

    df = pd.DataFrame(out_rows, columns=cols) if out_rows else pd.DataFrame(columns=cols)
    return df.rename(columns=FINAL_COLUMN_HEADERS)


def build_detailed_model_outputs_df(clean_rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Tạo bảng `detailed_model_outputs.csv` (tiêu đề song ngữ 2 dòng Anh–Việt, đồng bộ STT & likes giảm dần với final_results.csv)."""
    norm_rows = _sort_rows_by_likes_desc([_strip_row_keys(r) for r in clean_rows])
    has_gemma = _has_gemma_audit_rows(norm_rows)

    def _is_vi(r: dict[str, Any]) -> bool:
        return str(r.get("language", "")).strip().lower() in {"vi", "tiếng việt"}

    def _is_en(r: dict[str, Any]) -> bool:
        return str(r.get("language", "")).strip().lower() in {"en", "tiếng anh"}

    has_vi = any(
        _is_vi(r) or bool(str(r.get("pred_phobert", "")).strip()) or bool(str(r.get("pred_visobert", "")).strip())
        for r in norm_rows
    )
    has_en = any(
        _is_en(r) or bool(str(r.get("pred_roberta", "")).strip())
        for r in norm_rows
    )

    has_dl = any(
        bool(str(r.get("dl_primary_model", "")).strip())
        or bool(str(r.get("pred_phobert", "")).strip())
        or bool(str(r.get("pred_roberta", "")).strip())
        or bool(str(r.get("pred_visobert", "")).strip())
        or bool(str(r.get("pred_dl_primary", "")).strip())
        for r in norm_rows
    )

    cols: list[str] = []
    for col in _DETAILED_CORE_COLUMNS:
        if col == "dl_primary_model":
            if has_dl:
                cols.append(col)
        elif col in {"pred_phobert", "pred_visobert"}:
            if has_dl and (has_vi or (not has_vi and not has_en)):
                cols.append(col)
        elif col == "pred_roberta":
            if has_dl and (has_en or (not has_vi and not has_en)):
                cols.append(col)
        else:
            cols.append(col)

    if has_gemma:
        cols.extend(_DETAILED_GEMMA_COLUMNS)

    out_rows: list[dict[str, Any]] = []
    for stt, r in enumerate(norm_rows, start=1):
        row_copy = dict(r)
        row_copy["stt"] = stt
        row_copy["likes"] = _safe_int(row_copy.get("likes"), 0)

        lang_raw = str(row_copy.get("language", "")).strip().lower()
        is_vi_row = lang_raw in {"vi", "tiếng việt"}
        is_en_row = lang_raw in {"en", "tiếng anh"}
        row_copy["language"] = _LANG_VI_MAP.get(lang_raw, row_copy.get("language", ""))

        dl_primary_fallback = str(row_copy.get("pred_dl_primary", "")).strip()
        if not str(row_copy.get("dl_primary_model", "")).strip():
            row_copy["dl_primary_model"] = "PhoBERT" if is_vi_row else ("Twitter-RoBERTa" if is_en_row else "")

        if is_vi_row:
            row_copy["pred_phobert"] = str(row_copy.get("pred_phobert", "")).strip() or dl_primary_fallback or "—"
            row_copy["pred_visobert"] = str(row_copy.get("pred_visobert", "")).strip() or "—"
            row_copy["pred_roberta"] = "—"
        elif is_en_row:
            row_copy["pred_phobert"] = "—"
            row_copy["pred_visobert"] = "—"
            row_copy["pred_roberta"] = str(row_copy.get("pred_roberta", "")).strip() or dl_primary_fallback or "—"
        else:
            for k in ("pred_phobert", "pred_visobert", "pred_roberta"):
                row_copy[k] = str(row_copy.get(k, "")).strip() or "—"

        out_rows.append({c: row_copy.get(c, "") for c in cols})

    df = pd.DataFrame(out_rows, columns=cols) if out_rows else pd.DataFrame(columns=cols)
    return df.rename(columns=DETAILED_COLUMN_HEADERS)


def resolve_local_input_dir(input_dir: str | Path) -> dict[str, Any]:
    """Tự động dò tìm file comment, video/ảnh và caption trong một thư mục dữ liệu cục bộ."""
    folder = Path(input_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục đầu vào: {folder}")

    comments_source: Path | None = None
    priority_names = (
        "raw_comments.csv",
        "comments.csv",
        "comments.json",
    )
    for fname in priority_names:
        candidate = folder / fname
        if candidate.is_file():
            comments_source = candidate
            break

    if comments_source is None:
        comment_exts = {".csv", ".json"}
        ignored_json_names = {"metadata.json", MEDIA_CONTEXT_FILENAME}
        for p in sorted(folder.iterdir()):
            if (
                p.is_file()
                and p.suffix.lower() in comment_exts
                and p.name.lower() not in ignored_json_names
            ):
                comments_source = p
                break

    media_input: Path | None = None
    video_mp4 = folder / "video.mp4"
    if video_mp4.is_file():
        media_input = video_mp4
    else:
        video_files = sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )
        if video_files:
            media_input = video_files[0]
        else:
            images_sub = folder / "images"
            if images_sub.is_dir() and any(
                p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in images_sub.iterdir()
            ):
                media_input = images_sub
            elif any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in folder.iterdir()):
                media_input = folder

    caption = ""
    meta_file = folder / "metadata.json"
    if meta_file.is_file():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                caption = str(meta.get("caption") or "").strip()
        except Exception:
            pass

    media_ctx_file = folder / MEDIA_CONTEXT_FILENAME
    media_context_path = media_ctx_file if media_ctx_file.is_file() else None

    if comments_source is None and media_input is None and media_context_path is None:
        raise ValueError(f"Thư mục `{folder}` không chứa file comment, video/ảnh hay media_context.json hợp lệ.")

    return {
        "comments_source": str(comments_source) if comments_source else None,
        "media_input": str(media_input) if media_input else None,
        "media_context_path": str(media_context_path) if media_context_path else None,
        "caption": caption,
        "run_name": folder.name,
    }


def _infer_run_name(
    tiktok_url: str | None,
    input_dir: str | Path | None = None,
) -> str:
    if tiktok_url:
        return extract_post_id(tiktok_url)
    if input_dir is not None:
        return Path(input_dir).name or "run"
    return "run"


def normalize_engine_mode(value: str | int | None) -> str:
    """Chuẩn hóa lựa chọn chế độ Giai đoạn 3: '1'/'classical' (ML truyền thống) hoặc '2'/'hybrid' (hiện tại)."""
    if value is None or str(value).strip() == "":
        return "hybrid"
    norm = str(value).strip().lower()
    if norm in {"1", "classical", "ml", "traditional"}:
        return "classical"
    if norm in {"2", "hybrid", "full"}:
        return "hybrid"
    raise ValueError(
        f"engine_mode không hợp lệ: {value!r}. Chỉ hỗ trợ '1'/'classical' hoặc '2'/'hybrid'."
    )


class SentimentPipeline:
    """Pipeline 4 giai đoạn: data/<post_id>/ (thu thập) -> outputs/<post_id>/ (kết quả sau xử lý)."""

    def __init__(
        self,
        media_analyzer: MediaAnalyzer | None = None,
        text_preprocessor: TextPreprocessor | None = None,
        comment_analyzer: CommentAnalyzer | None = None,
        ai_auditor: GemmaAuditor | None = None,
    ) -> None:
        self.media_analyzer = media_analyzer or MediaAnalyzer()
        self.text_preprocessor = text_preprocessor or TextPreprocessor()
        self.comment_analyzer = comment_analyzer or CommentAnalyzer()
        self.ai_auditor = ai_auditor or GemmaAuditor()

    @staticmethod
    def _build_noise_summary(preprocessed: Sequence[PreprocessedComment]) -> dict[str, Any]:
        total = len(preprocessed)
        valid_cnt = sum(1 for p in preprocessed if p.is_valid)
        noise_items = [p for p in preprocessed if not p.is_valid]
        dup_cnt = sum(1 for p in preprocessed if p.is_valid and p.is_duplicate)
        reason_counts = dict(Counter(p.filter_reason for p in noise_items))
        return {
            "total_raw_comments": total,
            "valid_comments": valid_cnt,
            "filtered_noise_count": len(noise_items),
            "duplicate_valid_count": dup_cnt,
            "filter_reasons": reason_counts,
        }

    @staticmethod
    def _export_outputs(
        out_path: Path,
        report: dict[str, Any],
        preprocessed: Sequence[PreprocessedComment],
        metadata_rows: Sequence[dict[str, Any]],
        analyzed: Sequence[AnalyzedComment],
        audits: Sequence[GemmaAuditItem],
    ) -> dict[str, str]:
        out_path.mkdir(parents=True, exist_ok=True)
        exported: dict[str, str] = {}

        # 1. report.json
        report_file = out_path / "report.json"
        with report_file.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        exported["report_json"] = str(report_file)

        # Chuẩn bị toàn bộ thuộc tính của từng comment sạch
        audit_map = {g.comment_index: g for g in audits}
        clean_rows: list[dict[str, Any]] = []
        for idx, item in enumerate(analyzed):
            g = audit_map.get(idx)
            clean_rows.append(
                {
                    "comment_id": item.comment_id,
                    "is_reply": item.is_reply,
                    "parent_comment_id": item.parent_comment_id,
                    "username": item.username,
                    "nickname": item.nickname,
                    "likes": item.likes,
                    "reply_count": item.reply_count,
                    "created_at": item.created_at,
                    "language": item.language,
                    "original_text": item.original_text,
                    "cleaned_text": item.cleaned_text,
                    "segmented_text": item.segmented_text,
                    "is_duplicate": item.is_duplicate,
                    "has_sarcasm_signal": item.has_sarcasm_signal,
                    "needs_audit": item.needs_audit,
                    "is_on_topic": item.is_on_topic,
                    "context_relevance": item.context_relevance,
                    "matched_video_terms": ", ".join(item.matched_video_terms),
                    "context_alignment": item.context_alignment,
                    "polarity_label": item.polarity_label,
                    "polarity_vi": item.polarity_vi,
                    "polarity_confidence": item.polarity_confidence,
                    "polarity_score": item.polarity_score,
                    "score_POS": item.polarity_probs.get("POS", 0.0),
                    "score_NEG": item.polarity_probs.get("NEG", 0.0),
                    "score_NEU": item.polarity_probs.get("NEU", 0.0),
                    "emotion_label": item.emotion_label,
                    "emotion_vi": item.emotion_vi,
                    "emotion_confidence": item.emotion_confidence,
                    "pred_nb": item.pred_nb,
                    "pred_cnb": item.pred_cnb,
                    "pred_lr": item.pred_lr,
                    "pred_ridge": item.pred_ridge,
                    "pred_sgd": item.pred_sgd,
                    "pred_svm": item.pred_svm,
                    "pred_nbsvm": item.pred_nbsvm,
                    "pred_rbf_svm": item.pred_rbf_svm,
                    "pred_rf": item.pred_rf,
                    "pred_tree": item.pred_tree,
                    "pred_ml_ensemble": item.pred_ml_ensemble,
                    "pred_lexicon": item.pred_lexicon,
                    "dl_primary_model": item.dl_primary_model,
                    "pred_phobert": item.pred_phobert,
                    "pred_visobert": item.pred_visobert,
                    "pred_roberta": item.pred_roberta,
                    "pred_dl_primary": item.pred_dl_primary,
                    "emotion_ml_label": item.emotion_ml_label,
                    "pred_emo_cnb": item.pred_emo_cnb,
                    "pred_emo_lr": item.pred_emo_lr,
                    "pred_emo_ridge": item.pred_emo_ridge,
                    "pred_emo_svm": item.pred_emo_svm,
                    "pred_emo_nbsvm": item.pred_emo_nbsvm,
                    "pred_emo_sgd": item.pred_emo_sgd,
                    "gemma_audited": g.gemma_audited if g else False,
                    "gemma_polarity": g.gemma_polarity if g else "",
                    "gemma_emotion": g.gemma_emotion if g else "",
                    "gemma_emotion_vi": g.gemma_emotion_vi if g else "",
                    "gemma_is_sarcasm": g.gemma_is_sarcasm if (g and g.gemma_audited) else "",
                    "gemma_agrees_polarity": (
                        g.gemma_agrees_polarity if (g and g.gemma_audited) else ""
                    ),
                    "gemma_agrees_emotion": (
                        g.gemma_agrees_emotion if (g and g.gemma_audited) else ""
                    ),
                    "gemma_reason": g.gemma_reason if g else "",
                }
            )

        # 2. final_results.csv (các thuộc tính phân tích hữu ích, sắp theo likes giảm dần)
        final_csv = out_path / "final_results.csv"
        format_final_results_df(clean_rows).to_csv(final_csv, index=False, encoding="utf-8-sig")
        exported["final_results_csv"] = str(final_csv)

        # 2b. detailed_model_outputs.csv (thuộc tính kỹ thuật, ngữ cảnh video, xác suất & dự đoán từng mô hình)
        detailed_csv = out_path / "detailed_model_outputs.csv"
        build_detailed_model_outputs_df(clean_rows).to_csv(
            detailed_csv, index=False, encoding="utf-8-sig"
        )
        exported["detailed_model_outputs_csv"] = str(detailed_csv)

        # 3. filtered_noise.csv (TÁCH RIÊNG comment nhiễu / ngôn ngữ khác — tiêu đề song ngữ 2 dòng Anh–Việt, sắp theo lượt thích giảm dần)
        noise_csv = out_path / "filtered_noise.csv"
        raw_noise_items: list[dict[str, Any]] = []
        for idx, p in enumerate(preprocessed):
            if p.is_valid:
                continue
            meta = metadata_rows[idx] if idx < len(metadata_rows) else {}
            lang_code = str(p.language or "").strip().lower()
            raw_noise_items.append(
                {
                    "original_text": p.original_text,
                    "filter_reason_vi": FILTER_REASON_VI_MAP.get(p.filter_reason, p.filter_reason),
                    "filter_reason": p.filter_reason,
                    "language": _LANG_VI_MAP.get(lang_code, lang_code),
                    "likes": _safe_int(meta.get("likes"), 0),
                    "username": str(meta.get("username", "")),
                    "nickname": str(meta.get("nickname", "")),
                    "comment_id": str(meta.get("comment_id", "")),
                }
            )
        raw_noise_items.sort(key=lambda r: r["likes"], reverse=True)
        noise_rows = [
            {
                "stt": stt,
                "original_text": r["original_text"],
                "filter_reason_vi": r["filter_reason_vi"],
                "filter_reason": r["filter_reason"],
                "language": r["language"],
                "likes": r["likes"],
                "username": r["username"],
                "nickname": r["nickname"],
                "comment_id": r["comment_id"],
            }
            for stt, r in enumerate(raw_noise_items, start=1)
        ]
        noise_cols = list(_NOISE_COLUMNS)
        df_noise = (
            pd.DataFrame(noise_rows, columns=noise_cols)
            if noise_rows
            else pd.DataFrame(columns=noise_cols)
        )
        df_noise.rename(columns=NOISE_COLUMN_HEADERS).to_csv(
            noise_csv, index=False, encoding="utf-8-sig"
        )
        exported["filtered_noise_csv"] = str(noise_csv)

        # 4. model_comparison.csv (tiêu đề song ngữ 2 dòng Anh–Việt, số nguyên sạch & ký hiệu "—" cho ô không áp dụng)
        comp_csv = out_path / "model_comparison.csv"
        comp_data = report.get("model_comparison") or {}
        has_gemma = _has_gemma_audit_rows(clean_rows)

        def _fmt_int_cell(val: Any) -> Any:
            if val is None or val == "" or pd.isna(val):
                return "—"
            return int(round(float(val)))

        def _fmt_pct_cell(val: Any) -> str:
            if val is None or val == "" or pd.isna(val):
                return "—"
            return f"{float(val):.2f}%"

        def _fmt_float_cell(val: Any, digits: int = 4) -> Any:
            if val is None or val == "" or pd.isna(val):
                return "—"
            return round(float(val), digits)

        comp_raw_items: list[tuple[str, dict[str, Any]]] = []
        for row in comp_data.get("polarity_models") or []:
            comp_raw_items.append(("Polarity (3 lớp)", row))
        for row in comp_data.get("emotion_models") or []:
            comp_raw_items.append(("Emotion (7 cung bậc)", row))

        comp_cols = [
            "task",
            "model",
            "family",
            "total_comments",
            "pos_count",
            "neu_count",
            "neg_count",
            "pos_pct",
            "neu_pct",
            "neg_pct",
        ]
        if has_gemma:
            comp_cols.extend(["agreement_vs_gemma_pct", "macro_f1_vs_gemma", "cohen_kappa_vs_gemma"])
        comp_cols.append("inference_time_sec")

        comp_formatted_rows: list[dict[str, Any]] = []
        for task_label, row in comp_raw_items:
            item_out: dict[str, Any] = {
                "task": task_label,
                "model": row.get("model", ""),
                "family": row.get("family", ""),
                "total_comments": _fmt_int_cell(row.get("total_comments")),
                "pos_count": _fmt_int_cell(row.get("pos_count")),
                "neu_count": _fmt_int_cell(row.get("neu_count")),
                "neg_count": _fmt_int_cell(row.get("neg_count")),
                "pos_pct": _fmt_pct_cell(row.get("pos_pct")),
                "neu_pct": _fmt_pct_cell(row.get("neu_pct")),
                "neg_pct": _fmt_pct_cell(row.get("neg_pct")),
                "agreement_vs_gemma_pct": _fmt_pct_cell(row.get("agreement_vs_gemma_pct")),
                "macro_f1_vs_gemma": _fmt_float_cell(row.get("macro_f1_vs_gemma"), 4),
                "cohen_kappa_vs_gemma": _fmt_float_cell(row.get("cohen_kappa_vs_gemma"), 4),
                "inference_time_sec": _fmt_float_cell(row.get("inference_time_sec"), 3),
            }
            comp_formatted_rows.append({c: item_out[c] for c in comp_cols})

        df_comp = (
            pd.DataFrame(comp_formatted_rows, columns=comp_cols)
            if comp_formatted_rows
            else pd.DataFrame(columns=comp_cols)
        )
        df_comp.rename(columns=COMPARISON_COLUMN_HEADERS).to_csv(
            comp_csv, index=False, encoding="utf-8-sig"
        )
        exported["model_comparison_csv"] = str(comp_csv)

        return exported

    def run(
        self,
        tiktok_url: str | None = None,
        input_dir: str | Path | None = None,
        enable_audio: bool = True,
        run_media_analysis: bool = True,
        force_reanalyze_media: bool = False,
        prepare_only: bool = False,
        unload_ollama_after_stage1: bool = True,
        run_ai_audit: bool = False,
        audit_limit: int | None = None,
        audit_scope: str = "all",
        engine_mode: str | int | None = None,
        max_comments: int = 200,
        include_replies: bool = True,
        max_replies_per_comment: int | None = None,
        crawl_headless: bool = False,
        crawl_interactive_pause: bool = True,
        deduplicate_stats: bool = False,
        data_dir: str | Path = "data",
        output_dir: str | Path | None = "outputs",
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Chạy pipeline: Chuẩn bị dữ liệu vào data/<post_id>/ -> Giai đoạn 2 & 3 -> (Tùy chọn Giai đoạn 4) -> outputs/<post_id>/."""
        if not tiktok_url and input_dir is None:
            raise ValueError(
                "Cần cung cấp `tiktok_url` hoặc `input_dir` (thư mục data/<post_id>/)."
            )
        if engine_mode is not None and str(engine_mode).strip() != "":
            self.comment_analyzer.engine_mode = normalize_engine_mode(engine_mode)

        data_root = Path(data_dir)
        effective_media: str | Path | Sequence[str | Path] | None = None
        effective_caption = ""
        effective_comments: str | Path | None = None
        preexisting_media_context_path: Path | None = None

        if input_dir is not None:
            resolved_dir = resolve_local_input_dir(input_dir)
            effective_comments = resolved_dir["comments_source"]
            effective_media = resolved_dir["media_input"]
            if resolved_dir.get("media_context_path"):
                preexisting_media_context_path = Path(resolved_dir["media_context_path"])
            effective_caption = resolved_dir["caption"]

        run_name = _infer_run_name(
            tiktok_url,
            input_dir=input_dir,
        )

        out_path: Path | None = None
        if output_dir is not None:
            base_out = Path(output_dir)
            out_path = base_out / run_name if base_out.name in {"outputs", "output"} else base_out

        tiktok_bundle: TikTokMediaBundle | None = None

        # CHUẨN BỊ DỮ LIỆU: Toàn bộ dữ liệu thu thập thô & ngữ cảnh media được lưu vào data/<post_id>/
        if tiktok_url:
            print(f"[Chuẩn bị dữ liệu] Đang tải video/ảnh & metadata vào {data_root / run_name}...")
            tiktok_bundle, crawled_csv = collect_tiktok_post(
                url=tiktok_url,
                data_dir=data_root,
                collect_comments=(effective_comments is None),
                max_comments=max_comments,
                include_replies=include_replies,
                max_replies_per_comment=max_replies_per_comment,
                headless=crawl_headless,
                interactive_pause=crawl_interactive_pause,
            )
            run_name = tiktok_bundle.post_id
            if output_dir is not None:
                base_out = Path(output_dir)
                out_path = base_out / run_name if base_out.name in {"outputs", "output"} else base_out

            if effective_media is None:
                effective_media = (
                    tiktok_bundle.media_paths[0]
                    if tiktok_bundle.media_type == "video"
                    else tiktok_bundle.media_paths
                )
            if not effective_caption.strip():
                effective_caption = tiktok_bundle.caption

            if effective_comments is None and crawled_csv is not None:
                effective_comments = crawled_csv

        resolved_data_dir: Path | None = None
        if input_dir is not None:
            resolved_data_dir = Path(input_dir)
        elif tiktok_url or effective_media is not None:
            resolved_data_dir = data_root / run_name

        media_context_path: Path | None = (
            preexisting_media_context_path
            if preexisting_media_context_path is not None
            else (resolved_data_dir / MEDIA_CONTEXT_FILENAME if resolved_data_dir is not None else None)
        )

        # GIAI ĐOẠN 1 (CHUẨN BỊ NGỮ CẢNH MEDIA -> LƯU / NẠP TỪ data/<post_id>/media_context.json)
        media_result: MediaAnalysisResult | None = None
        ran_stage1_inference = False
        if run_media_analysis:
            if media_context_path is not None and media_context_path.is_file() and not force_reanalyze_media:
                print(
                    f"[Chuẩn bị dữ liệu] Đã nạp nội dung video/ảnh có sẵn từ {media_context_path} "
                    "(bỏ qua chạy lại Gemma 4)."
                )
                media_result = MediaAnalysisResult.load_json(media_context_path)
            elif effective_media is not None:
                print("[Giai đoạn 1] Gemma 4 E2B đang phân tích ngữ cảnh video/ảnh...")
                media_result = self.media_analyzer.analyze(
                    media_input=effective_media,
                    caption=effective_caption,
                    enable_audio=enable_audio,
                )
                ran_stage1_inference = True
                if media_context_path is not None:
                    saved_ctx = media_result.save_json(media_context_path)
                    print(f"[Chuẩn bị dữ liệu] Đã lưu ngữ cảnh video/ảnh vào {saved_ctx}")
            else:
                print("[Giai đoạn 1] Không có media đầu vào -> chạy thẳng trên bình luận.")
        elif effective_media is not None:
            print("[Giai đoạn 1] Bỏ qua phân tích media (run_media_analysis=False).")
        else:
            print("[Giai đoạn 1] Không có media đầu vào -> chạy thẳng trên bình luận.")

        if unload_ollama_after_stage1 and ran_stage1_inference:
            self.media_analyzer.unload_ollama_model()
            self.media_analyzer.unload_whisper_model()

        if prepare_only:
            print(
                f"[Chuẩn bị dữ liệu] Hoàn tất chuẩn bị gói dữ liệu tại "
                f"{resolved_data_dir or (data_root / run_name)} (prepare_only=True)."
            )
            return {
                "run_name": run_name,
                "prepare_only": True,
                "data_dir": str(resolved_data_dir) if resolved_data_dir else None,
                "media_context_path": (
                    str(media_context_path)
                    if (media_context_path is not None and media_context_path.is_file())
                    else None
                ),
                "tiktok_metadata": tiktok_bundle.to_dict() if tiktok_bundle else None,
                "media_analysis": media_result.to_dict() if media_result else None,
            }

        # GIAI ĐOẠN 2 & 3 (NLP + ML + DL SONG NGỮ VI/EN CÓ NGỮ CẢNH VIDEO)
        preprocessed: list[PreprocessedComment] = []
        metadata_rows: list[dict[str, Any]] = []
        analyzed: list[AnalyzedComment] = []
        timings: dict[str, float] = {}
        summary: dict[str, Any] | None = None
        keywords_tfidf: dict[str, list[dict[str, Any]]] = {}

        if effective_comments is not None:
            df_comments = load_comments_dataframe(effective_comments)
            metadata_rows = df_comments.to_dict(orient="records")
            raw_texts = df_comments["comment"].tolist()

            print(f"[Giai đoạn 2] Lọc nhiễu, nhận diện ngôn ngữ (VI/EN) & đối chiếu ngữ cảnh ({len(raw_texts)} bình luận)...")
            preprocessed = self.text_preprocessor.process_batch(
                raw_texts,
                media_context=media_result,
            )
            noise_summary = self._build_noise_summary(preprocessed)
            print(
                f"  -> Hợp lệ: {noise_summary['valid_comments']} | "
                f"Đã lọc nhiễu: {noise_summary['filtered_noise_count']}"
            )

            mode_label = (
                "Classical ML v3 thuần túy (0% Deep Learning)"
                if getattr(self.comment_analyzer, "engine_mode", "hybrid") == "classical"
                else "Hybrid (5 họ Classical ML v3 + Lexicon + Deep Learning VI/EN)"
            )
            print(f"[Giai đoạn 3] Đang chạy {mode_label} & Context Alignment...")
            analyzed, timings = self.comment_analyzer.analyze_comments(
                preprocessed=preprocessed,
                metadata_rows=metadata_rows,
                media_result=media_result,
                show_progress=show_progress,
            )
            summary = self.comment_analyzer.summarize(
                analyzed, deduplicate_stats=deduplicate_stats
            )

            valid_pre = [p for p in preprocessed if p.is_valid]
            valid_pols = [a.polarity_label for a in analyzed]
            keywords_tfidf = extract_keywords_tfidf(valid_pre, valid_pols, top_k=10)
        else:
            noise_summary = self._build_noise_summary([])

        # GIAI ĐOẠN 4 (AI ĐẦU RA — TÙY CHỌN)
        audits: list[GemmaAuditItem] = []
        ai_verdict: dict[str, Any] | None = None
        scope_norm = str(audit_scope or "all").strip().lower()
        if scope_norm not in {"all", "uncertain"}:
            scope_norm = "all"

        ai_audit_status: dict[str, Any] = {
            "requested": bool(run_ai_audit),
            "scope": scope_norm,
            "ollama_ready": None,
            "audited_count": 0,
            "total_comments": len(analyzed),
            "message": "",
        }
        if analyzed and run_ai_audit:
            ok, reason = self.ai_auditor.check_ollama_ready()
            ai_audit_status["ollama_ready"] = ok
            if not ok:
                print(f"[Cảnh báo Giai đoạn 4] {reason}")
                ai_audit_status["message"] = reason
            else:
                print(
                    f"[Giai đoạn 4] Gemma 4 E2B đang thẩm định thực tế bình luận theo ngữ cảnh "
                    f"(phạm vi: {scope_norm})..."
                )
                t_aud = time.perf_counter()
                audits = self.ai_auditor.audit_comments(
                    analyzed=analyzed,
                    media_result=media_result,
                    audit_limit=audit_limit,
                    audit_scope=scope_norm,
                    show_progress=show_progress,
                )
                timings["gemma_audit_sec"] = round(time.perf_counter() - t_aud, 3)
                audited_cnt = sum(1 for g in audits if g.gemma_audited)
                ai_audit_status["audited_count"] = audited_cnt
                ai_audit_status["message"] = (
                    f"Đã thẩm định {audited_cnt}/{len(analyzed)} bình luận (phạm vi: {scope_norm})."
                )
        elif analyzed and not run_ai_audit:
            ai_audit_status["message"] = (
                "Đã tắt Giai đoạn 4 theo lựa chọn người dùng (run_ai_audit=False)."
            )
            print("[Giai đoạn 4] Đã bỏ qua thẩm định AI theo tùy chọn người dùng (run_ai_audit=False).")
        else:
            ai_audit_status["message"] = "Không có bình luận hợp lệ để thẩm định."

        model_comparison = build_model_comparison(
            analyzed=analyzed,
            audits=audits,
            timings=timings,
        )

        if analyzed and run_ai_audit and any(g.gemma_audited for g in audits):
            ai_verdict = self.ai_auditor.generate_overall_verdict(
                analyzed=analyzed,
                audits=audits,
                model_comparison=model_comparison.get("polarity_models") or [],
                media_result=media_result,
            )

        report: dict[str, Any] = {
            "run_name": run_name,
            "tiktok_metadata": tiktok_bundle.to_dict() if tiktok_bundle else None,
            "media_analysis": media_result.to_dict() if media_result else None,
            "noise_filter_summary": noise_summary,
            "keywords_tfidf": keywords_tfidf,
            "sentiment_summary": summary,
            "model_comparison": model_comparison,
            "ai_audit_verdict": ai_verdict,
            "ai_audit_status": ai_audit_status,
        }

        if out_path is not None:
            exported = self._export_outputs(
                out_path=out_path,
                report=report,
                preprocessed=preprocessed,
                metadata_rows=metadata_rows,
                analyzed=analyzed,
                audits=audits,
            )
            if summary is not None and summary.get("total_valid_comments", 0) > 0:
                dash_path = out_path / "dashboard.png"
                fig = plot_sentiment_dashboard(report, save_path=dash_path)
                plt.close(fig)
                exported["dashboard_png"] = str(dash_path)
            report["exported_files"] = exported

        return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hệ thống Sentiment Analysis TikTok Song ngữ (VI/EN): Gemma 4 E2B + ML/DL."
    )
    parser.add_argument("--url", default=None, help="Link video/bài viết TikTok.")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Thư mục cục bộ chứa sẵn file comment/media/media_context.json (vd: data/<post_id>).",
    )
    parser.add_argument("--no-audio", action="store_true", help="Bỏ qua tách và nhận dạng giọng nói.")
    parser.add_argument(
        "--skip-media",
        action="store_true",
        help="Bỏ qua Giai đoạn 1 (không chạy Gemma 4 đọc video/ảnh).",
    )
    parser.add_argument(
        "--reanalyze-media",
        action="store_true",
        help="Buộc chạy lại Gemma 4 + Whisper ngay cả khi đã có sẵn data/<post_id>/media_context.json.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Chỉ chạy bước Chuẩn bị dữ liệu (lưu media, comments, media_context.json vào data/<post_id>/) rồi dừng.",
    )
    audit_group = parser.add_mutually_exclusive_group()
    audit_group.add_argument(
        "--ai-audit",
        dest="run_ai_audit",
        action="store_true",
        help="Bật Giai đoạn 4 (dùng Gemma 4 E2B thẩm định các ca mỉa mai sâu & kết quả Giai đoạn 3).",
    )
    audit_group.add_argument(
        "--no-ai-audit",
        "--skip-audit",
        dest="run_ai_audit",
        action="store_false",
        help="Tắt Giai đoạn 4 (mặc định tắt — bỏ qua bước kiểm tra bằng Gemma 4 E2B).",
    )
    parser.set_defaults(run_ai_audit=False)
    parser.add_argument("--audit-limit", type=int, default=None, help="Giới hạn số comment thẩm định.")
    parser.add_argument(
        "--audit-scope",
        default="all",
        choices=["all", "uncertain"],
        help="Phạm vi Giai đoạn 4: all (toàn bộ bình luận hợp lệ) hoặc uncertain (chỉ các ca khó needs_audit=True).",
    )
    parser.add_argument("--max-comments", type=int, default=200, help="Số comment tối đa khi crawl.")
    parser.add_argument("--no-replies", action="store_true", help="Không thu thập bình luận trả lời (reply).")
    parser.add_argument(
        "--max-replies-per-comment",
        type=int,
        default=None,
        help="Giới hạn số reply tối đa cho mỗi bình luận chính (mặc định: không giới hạn).",
    )
    parser.add_argument("--headless-crawl", action="store_true", help="Crawl ẩn trình duyệt.")
    parser.add_argument("--no-crawl-pause", action="store_true", help="Không tạm dừng nhấn Enter.")
    parser.add_argument("--deduplicate-stats", action="store_true", help="Loại trùng lặp khỏi thống kê.")
    parser.add_argument(
        "--engine",
        default="hybrid",
        choices=["hybrid", "classical", "1", "2"],
        help="Chế độ Giai đoạn 3: 1/classical (ML truyền thống) hoặc 2/hybrid (như hiện tại: ML v3 + Deep Learning).",
    )
    parser.add_argument(
        "--retrain-ml",
        action="store_true",
        help="Huấn luyện lại cụm mô hình Classical ML v3 (VI & EN) trước khi chạy.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="gemma4-e2b-opt")
    parser.add_argument("--visobert-model", default=DEFAULT_VISOBERT_MODEL)
    parser.add_argument("--phobert-model", default=DEFAULT_PHOBERT_SENTIMENT_MODEL)
    parser.add_argument("--emotion-model", default=DEFAULT_PHOBERT_EMOTION_MODEL)
    parser.add_argument("--en-sentiment-model", default=DEFAULT_EN_SENTIMENT_MODEL)
    parser.add_argument("--en-emotion-model", default=DEFAULT_EN_EMOTION_MODEL)
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--data-dir", default="data", help="Thư mục lưu dữ liệu thô thu thập.")
    parser.add_argument("--output", default="outputs", help="Thư mục lưu kết quả cuối sau xử lý.")
    parser.add_argument("--save-chart", default=None, help="Đường dẫn lưu riêng file dashboard (tùy chọn).")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    engine_mode = normalize_engine_mode(args.engine)
    analyzer = MediaAnalyzer(
        ollama_url=args.ollama_url,
        model_name=args.ollama_model,
        whisper_model=args.whisper_model,
    )
    comment_analyzer = CommentAnalyzer(
        visobert_model=args.visobert_model,
        phobert_sentiment_model=args.phobert_model,
        phobert_emotion_model=args.emotion_model,
        en_sentiment_model=args.en_sentiment_model,
        en_emotion_model=args.en_emotion_model,
        engine_mode=engine_mode,
        device=args.device,
    )
    if args.retrain_ml:
        comment_analyzer.classical_suite.load_or_train("vi", force_retrain=True)
        comment_analyzer.classical_suite.load_or_train("en", force_retrain=True)
    auditor = GemmaAuditor(
        ollama_url=args.ollama_url,
        model_name=args.ollama_model,
    )
    pipeline = SentimentPipeline(
        media_analyzer=analyzer,
        comment_analyzer=comment_analyzer,
        ai_auditor=auditor,
    )

    report = pipeline.run(
        tiktok_url=args.url,
        input_dir=args.input_dir,
        enable_audio=not args.no_audio,
        run_media_analysis=not args.skip_media,
        force_reanalyze_media=args.reanalyze_media,
        prepare_only=args.prepare_only,
        run_ai_audit=args.run_ai_audit,
        audit_limit=args.audit_limit,
        audit_scope=args.audit_scope,
        engine_mode=engine_mode,
        max_comments=args.max_comments,
        include_replies=not args.no_replies,
        max_replies_per_comment=args.max_replies_per_comment,
        crawl_headless=args.headless_crawl,
        crawl_interactive_pause=not args.no_crawl_pause,
        deduplicate_stats=args.deduplicate_stats,
        data_dir=args.data_dir,
        output_dir=args.output,
    )

    if report.get("prepare_only"):
        print("\n=== HOÀN TẤT CHUẨN BỊ DỮ LIỆU ===")
        print(f"- Thư mục dữ liệu : {report.get('data_dir')}")
        print(f"- Media context   : {report.get('media_context_path')}")
        return

    print("\n=== KẾT QUẢ PHÂN TÍCH ===")
    if report["media_analysis"]:
        m = report["media_analysis"]
        print(f"- Chủ đề chính : {m['chu_de_chinh']}")
        print(f"- Mục đích     : {m['muc_dich_chinh']}")
        print(f"- Quan điểm    : {m['quan_diem_noi_dung']}")
        print(f"- Sắc thái bài : {m['sac_thai_noi_dung']}")
        print(f"- Đối tượng    : {', '.join(m['doi_tuong_chinh'])}")
        print(f"- Tóm tắt      : {m['tom_tat_ngan']}")

    nf = report["noise_filter_summary"]
    print("\n=== LỌC NHIỄU BÌNH LUẬN ===")
    print(
        f"- Tổng thô: {nf['total_raw_comments']} | Hợp lệ: {nf['valid_comments']} | "
        f"Đã lọc: {nf['filtered_noise_count']} | Trùng lặp: {nf['duplicate_valid_count']}"
    )
    if nf["filter_reasons"]:
        print(f"- Chi tiết lý do lọc: {nf['filter_reasons']}")

    if report["sentiment_summary"]:
        s = report["sentiment_summary"]
        print("\n=== THỐNG KÊ ĐA TẦNG (SONG NGỮ VI/EN) ===")
        print(f"- Ngôn ngữ     : {s['language_counts']}")
        print(f"- Polarity     : {s['polarity_counts']} | Tỷ lệ (%): {s['polarity_percentages']}")
        print(f"- Xu hướng     : {s['overall_tendency']}")
        print(
            f"- Điểm cực tính TB: {s['avg_polarity_score']} [-1..+1] | "
            f"Độ tự tin TB: {s['avg_confidence']}"
        )
        print(f"- 7 Cảm xúc (%): {s['emotion_percentages']}")
        if s.get("on_topic_summary"):
            ot = s["on_topic_summary"]
            print(
                f"- Bám sát video : {ot['on_topic_count']}/{s['total_valid_comments']} "
                f"({ot['on_topic_pct']}%) | Trung bình liên quan: {ot['avg_context_relevance']} | "
                f"Polarity nhóm bám sát: {ot['on_topic_polarity_percentages']}"
            )

        if args.save_chart:
            fig = plot_sentiment_dashboard(report, save_path=args.save_chart)
            plt.close(fig)

    comp = report.get("model_comparison") or {}
    if comp.get("polarity_models"):
        print("\n=== SO SÁNH CÁC MÔ HÌNH ===")
        df_comp = pd.DataFrame(comp["polarity_models"])
        cols = [
            c
            for c in [
                "model",
                "family",
                "pos_pct",
                "neu_pct",
                "neg_pct",
                "agreement_vs_gemma_pct",
                "macro_f1_vs_gemma",
            ]
            if c in df_comp.columns and df_comp[c].notna().any()
        ]
        print(df_comp[cols].to_string(index=False))

    if report.get("ai_audit_verdict"):
        v = report["ai_audit_verdict"]
        print("\n=== THẨM ĐỊNH ĐẦU RA (GEMMA 4 E2B) ===")
        print(f"- Đánh giá chung : {v.get('accuracy_verdict')} | Tốt nhất: {v.get('best_model')}")
        print(
            f"- Tỷ lệ mỉa mai  : {v.get('sarcasm_rate_pct')}% "
            f"({v.get('sarcasm_detected_count')}/{v.get('audited_count')})"
        )
        print(f"- Nhận xét       : {v.get('summary_vi')}")
    elif (report.get("ai_audit_status") or {}).get("requested"):
        st = report["ai_audit_status"]
        print("\n=== THẨM ĐỊNH ĐẦU RA (GEMMA 4 E2B) ===")
        print(f"- Trạng thái     : Không thực hiện được ({st.get('message')})")

    if report.get("exported_files"):
        print("\n=== FILE ĐÃ XUẤT ===")
        for k, v in report["exported_files"].items():
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
