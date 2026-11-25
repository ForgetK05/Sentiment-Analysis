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
    crawl_tiktok_comments,
    download_tiktok_media,
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


def load_comments(
    comments_source: str | Path | Sequence[str],
    text_column: str | None = None,
) -> list[str]:
    """Đọc danh sách chuỗi bình luận từ CSV / Excel / JSON / TXT."""
    df = load_comments_dataframe(comments_source, text_column=text_column)
    return df["comment"].tolist()


_FINAL_CORE_COLUMNS: tuple[str, ...] = (
    "comment_id",
    "is_reply",
    "parent_comment_id",
    "username",
    "likes",
    "reply_count",
    "created_at",
    "language",
    "original_text",
    "cleaned_text",
    "polarity_label",
    "polarity_vi",
    "polarity_confidence",
    "score_5",
    "score_10",
    "rating_level_vi",
    "emotion_vi",
    "emotion_confidence",
)

_FINAL_GEMMA_COLUMNS: tuple[str, ...] = (
    "gemma_polarity",
    "gemma_emotion_vi",
    "gemma_score_5",
    "gemma_is_sarcasm",
    "gemma_agrees_polarity",
    "gemma_reason",
)

FINAL_COLUMN_HEADERS: dict[str, str] = {
    "comment_id": "comment_id\n(mã bình luận)",
    "is_reply": "is_reply\n(là bình luận trả lời)",
    "parent_comment_id": "parent_comment_id\n(mã bình luận gốc)",
    "username": "username\n(tên tài khoản)",
    "likes": "likes\n(lượt thích)",
    "reply_count": "reply_count\n(số phản hồi)",
    "created_at": "created_at\n(thời gian đăng)",
    "language": "language\n(ngôn ngữ)",
    "original_text": "original_text\n(nội dung gốc)",
    "cleaned_text": "cleaned_text\n(nội dung đã làm sạch)",
    "polarity_label": "polarity_label\n(nhãn cực tính)",
    "polarity_vi": "polarity_vi\n(cực tính tiếng việt)",
    "polarity_confidence": "polarity_confidence\n(độ tự tin cực tính)",
    "score_5": "score_5\n(điểm thang 1-5)",
    "score_10": "score_10\n(điểm thang 0-10)",
    "rating_level_vi": "rating_level_vi\n(mức độ đánh giá)",
    "emotion_vi": "emotion_vi\n(cảm xúc 7 cung bậc)",
    "emotion_confidence": "emotion_confidence\n(độ tự tin cảm xúc)",
    "gemma_polarity": "gemma_polarity\n(cực tính gemma)",
    "gemma_emotion_vi": "gemma_emotion_vi\n(cảm xúc gemma)",
    "gemma_score_5": "gemma_score_5\n(điểm 1-5 gemma)",
    "gemma_is_sarcasm": "gemma_is_sarcasm\n(nói mỉa theo gemma)",
    "gemma_agrees_polarity": "gemma_agrees_polarity\n(khớp cực tính với gemma)",
    "gemma_reason": "gemma_reason\n(lý do gemma giải thích)",
}

_CLEANED_CORE_COLUMNS: tuple[str, ...] = (
    "username",
    "is_reply",
    "likes",
    "created_at",
    "language",
    "original_text",
    "polarity_vi",
    "score_5",
    "score_10",
    "emotion_vi",
)

_CLEANED_GEMMA_COLUMNS: tuple[str, ...] = (
    "gemma_polarity_vi",
    "gemma_score_5",
    "gemma_is_sarcasm",
    "gemma_reason",
)

CLEANED_COLUMN_HEADERS: dict[str, str] = {
    "username": "username\n(tên tài khoản)",
    "is_reply": "is_reply\n(là bình luận trả lời)",
    "likes": "likes\n(lượt thích)",
    "created_at": "created_at\n(thời gian đăng)",
    "language": "language\n(ngôn ngữ)",
    "original_text": "original_text\n(nội dung gốc)",
    "polarity_vi": "polarity_vi\n(cực tính hệ thống)",
    "score_5": "score_5\n(điểm thang 1-5)",
    "score_10": "score_10\n(điểm thang 0-10)",
    "emotion_vi": "emotion_vi\n(cảm xúc 7 cung bậc)",
    "gemma_polarity_vi": "gemma_polarity_vi\n(cực tính gemma tiếng việt)",
    "gemma_score_5": "gemma_score_5\n(điểm 1-5 gemma)",
    "gemma_is_sarcasm": "gemma_is_sarcasm\n(nói mỉa theo gemma)",
    "gemma_reason": "gemma_reason\n(lý do gemma giải thích)",
}

_DETAILED_CORE_COLUMNS: tuple[str, ...] = (
    "comment_id",
    "username",
    "original_text",
    "nickname",
    "segmented_text",
    "is_duplicate",
    "has_sarcasm_signal",
    "needs_audit",
    "polarity_score",
    "score_POS",
    "score_NEG",
    "score_NEU",
    "emotion_label",
    "pred_nb",
    "pred_lr",
    "pred_svm",
    "pred_lexicon",
    "pred_visobert",
    "pred_dl_primary",
    "emotion_ml_label",
)

_DETAILED_GEMMA_COLUMNS: tuple[str, ...] = (
    "gemma_audited",
    "gemma_emotion",
    "gemma_agrees_emotion",
)

DETAILED_COLUMN_HEADERS: dict[str, str] = {
    "comment_id": "comment_id\n(mã bình luận)",
    "username": "username\n(tên tài khoản)",
    "original_text": "original_text\n(nội dung gốc)",
    "nickname": "nickname\n(tên hiển thị)",
    "segmented_text": "segmented_text\n(văn bản đã tách từ)",
    "is_duplicate": "is_duplicate\n(bình luận trùng lặp)",
    "has_sarcasm_signal": "has_sarcasm_signal\n(dấu hiệu mỉa mai)",
    "needs_audit": "needs_audit\n(cần thẩm định kỹ)",
    "polarity_score": "polarity_score\n(điểm cực tính -1 đến +1)",
    "score_POS": "score_POS\n(xác suất tích cực)",
    "score_NEG": "score_NEG\n(xác suất tiêu cực)",
    "score_NEU": "score_NEU\n(xác suất trung tính)",
    "emotion_label": "emotion_label\n(nhãn cảm xúc)",
    "pred_nb": "pred_nb\n(dự đoán naive bayes)",
    "pred_lr": "pred_lr\n(dự đoán logistic regression)",
    "pred_svm": "pred_svm\n(dự đoán linear svm)",
    "pred_lexicon": "pred_lexicon\n(dự đoán từ điển luật)",
    "pred_visobert": "pred_visobert\n(dự đoán visobert)",
    "pred_dl_primary": "pred_dl_primary\n(dự đoán học sâu chính)",
    "emotion_ml_label": "emotion_ml_label\n(cảm xúc ml cổ điển)",
    "gemma_audited": "gemma_audited\n(đã thẩm định gemma)",
    "gemma_emotion": "gemma_emotion\n(nhãn cảm xúc gemma)",
    "gemma_agrees_emotion": "gemma_agrees_emotion\n(khớp cảm xúc với gemma)",
}


def _strip_row_keys(row: dict[str, Any]) -> dict[str, Any]:
    """Chuẩn hóa key về tên cột tiếng Anh gốc và dọn giá trị NaN."""
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


def format_final_results_df(clean_rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Tạo DataFrame `final_results.csv` chỉ chứa các thuộc tính hữu ích (18 cột lõi, hoặc 24 cột khi bật AI Audit)."""
    norm_rows = [_strip_row_keys(r) for r in clean_rows]
    has_gemma = _has_gemma_audit_rows(norm_rows)
    cols = list(_FINAL_CORE_COLUMNS) + (list(_FINAL_GEMMA_COLUMNS) if has_gemma else [])
    df = pd.DataFrame(norm_rows, columns=cols) if norm_rows else pd.DataFrame(columns=cols)
    return df.rename(columns=FINAL_COLUMN_HEADERS)


def build_cleaned_final_results_df(clean_rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Tạo bản tóm gọn `cleaned_final_results.csv` (10 cột lõi hoặc 14 cột khi bật AI Audit, sắp theo likes giảm dần)."""
    norm_rows = [_strip_row_keys(r) for r in clean_rows]
    has_gemma = _has_gemma_audit_rows(norm_rows)
    selected_cols = list(_CLEANED_CORE_COLUMNS) + (list(_CLEANED_GEMMA_COLUMNS) if has_gemma else [])

    out_rows: list[dict[str, Any]] = []
    for r in norm_rows:
        g_pol = str(r.get("gemma_polarity", "")).strip()
        g_score_raw = r.get("gemma_score_5", "")
        if g_score_raw != "":
            try:
                g_score_fmt: Any = int(float(g_score_raw))
            except (TypeError, ValueError):
                g_score_fmt = g_score_raw
        else:
            g_score_fmt = ""

        item: dict[str, Any] = {
            "username": r.get("username", ""),
            "is_reply": r.get("is_reply", False),
            "likes": int(float(r.get("likes", 0) or 0)),
            "created_at": r.get("created_at", ""),
            "language": r.get("language", ""),
            "original_text": r.get("original_text", ""),
            "polarity_vi": r.get("polarity_vi", ""),
            "score_5": int(float(r.get("score_5", 3) or 3)),
            "score_10": float(r.get("score_10", 5.0) or 5.0),
            "emotion_vi": r.get("emotion_vi", ""),
            "gemma_polarity_vi": LABEL_VI_MAP.get(g_pol, g_pol),
            "gemma_score_5": g_score_fmt,
            "gemma_is_sarcasm": r.get("gemma_is_sarcasm", ""),
            "gemma_reason": r.get("gemma_reason", ""),
        }
        out_rows.append({c: item[c] for c in selected_cols})

    df = pd.DataFrame(out_rows, columns=selected_cols) if out_rows else pd.DataFrame(columns=selected_cols)
    if not df.empty and "likes" in df.columns:
        df = df.sort_values(by="likes", ascending=False, kind="mergesort").reset_index(drop=True)
    return df.rename(columns=CLEANED_COLUMN_HEADERS)


def build_detailed_model_outputs_df(clean_rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Tạo bảng `detailed_model_outputs.csv` chứa các thuộc tính kỹ thuật, xác suất và dự đoán từng mô hình."""
    norm_rows = [_strip_row_keys(r) for r in clean_rows]
    has_gemma = _has_gemma_audit_rows(norm_rows)
    cols = list(_DETAILED_CORE_COLUMNS) + (list(_DETAILED_GEMMA_COLUMNS) if has_gemma else [])
    df = pd.DataFrame(norm_rows, columns=cols) if norm_rows else pd.DataFrame(columns=cols)
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
        "comments.xlsx",
        "comments.xls",
        "comments.json",
        "comments.txt",
    )
    for fname in priority_names:
        candidate = folder / fname
        if candidate.is_file():
            comments_source = candidate
            break

    if comments_source is None:
        comment_exts = {".csv", ".xlsx", ".xls", ".json", ".txt"}
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in comment_exts and p.name.lower() != "metadata.json":
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

    if comments_source is None and media_input is None:
        raise ValueError(f"Thư mục `{folder}` không chứa file comment hay video/ảnh hợp lệ.")

    return {
        "comments_source": str(comments_source) if comments_source else None,
        "media_input": str(media_input) if media_input else None,
        "caption": caption,
        "run_name": folder.name,
    }


def _infer_run_name(
    tiktok_url: str | None,
    comments_source: str | Path | Sequence[str] | None,
    media_input: str | Path | Sequence[str | Path] | None,
    input_dir: str | Path | None = None,
) -> str:
    if tiktok_url:
        return extract_post_id(tiktok_url)
    if input_dir is not None:
        return Path(input_dir).name or "run"
    if isinstance(comments_source, (str, Path)):
        p = Path(comments_source)
        # Nếu nằm trong data/<post_id>/raw_comments.csv -> lấy <post_id>
        if p.stem.lower() in {"raw_comments", "comments"} and p.parent.name not in {"data", "samples", ""}:
            return p.parent.name
        return p.stem
    if isinstance(media_input, (str, Path)):
        p = Path(media_input)
        return p.parent.name if p.stem.lower() == "video" and p.parent.name else p.stem
    return "run"


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
                    "polarity_label": item.polarity_label,
                    "polarity_vi": item.polarity_vi,
                    "polarity_confidence": item.polarity_confidence,
                    "polarity_score": item.polarity_score,
                    "score_5": item.score_5,
                    "score_10": item.score_10,
                    "rating_level_vi": item.rating_level_vi,
                    "score_POS": item.polarity_probs.get("POS", 0.0),
                    "score_NEG": item.polarity_probs.get("NEG", 0.0),
                    "score_NEU": item.polarity_probs.get("NEU", 0.0),
                    "emotion_label": item.emotion_label,
                    "emotion_vi": item.emotion_vi,
                    "emotion_confidence": item.emotion_confidence,
                    "pred_nb": item.pred_nb,
                    "pred_lr": item.pred_lr,
                    "pred_svm": item.pred_svm,
                    "pred_lexicon": item.pred_lexicon,
                    "pred_visobert": item.pred_visobert,
                    "pred_dl_primary": item.pred_dl_primary,
                    "emotion_ml_label": item.emotion_ml_label,
                    "gemma_audited": g.gemma_audited if g else False,
                    "gemma_polarity": g.gemma_polarity if g else "",
                    "gemma_emotion": g.gemma_emotion if g else "",
                    "gemma_emotion_vi": g.gemma_emotion_vi if g else "",
                    "gemma_score_5": g.gemma_score_5 if (g and g.gemma_audited) else "",
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

        # 2. final_results.csv (CHỈ các thuộc tính hữu ích)
        final_csv = out_path / "final_results.csv"
        format_final_results_df(clean_rows).to_csv(final_csv, index=False, encoding="utf-8-sig")
        exported["final_results_csv"] = str(final_csv)

        # 2b. cleaned_final_results.csv (bản tóm gọn nhanh, sắp theo likes giảm dần)
        cleaned_csv = out_path / "cleaned_final_results.csv"
        build_cleaned_final_results_df(clean_rows).to_csv(
            cleaned_csv, index=False, encoding="utf-8-sig"
        )
        exported["cleaned_final_results_csv"] = str(cleaned_csv)

        # 2c. detailed_model_outputs.csv (thuộc tính kỹ thuật, xác suất & dự đoán từng mô hình)
        detailed_csv = out_path / "detailed_model_outputs.csv"
        build_detailed_model_outputs_df(clean_rows).to_csv(
            detailed_csv, index=False, encoding="utf-8-sig"
        )
        exported["detailed_model_outputs_csv"] = str(detailed_csv)

        # 3. filtered_noise.csv (TÁCH RIÊNG comment nhiễu / ngôn ngữ khác)
        noise_csv = out_path / "filtered_noise.csv"
        noise_rows: list[dict[str, Any]] = []
        for idx, p in enumerate(preprocessed):
            if p.is_valid:
                continue
            meta = metadata_rows[idx] if idx < len(metadata_rows) else {}
            noise_rows.append(
                {
                    "comment_id": str(meta.get("comment_id", "")),
                    "username": str(meta.get("username", "")),
                    "nickname": str(meta.get("nickname", "")),
                    "likes": int(meta.get("likes", 0) or 0),
                    "language": p.language,
                    "original_text": p.original_text,
                    "cleaned_text": p.cleaned_text,
                    "filter_reason": p.filter_reason,
                }
            )
        pd.DataFrame(
            noise_rows,
            columns=[
                "comment_id",
                "username",
                "nickname",
                "likes",
                "language",
                "original_text",
                "cleaned_text",
                "filter_reason",
            ],
        ).to_csv(noise_csv, index=False, encoding="utf-8-sig")
        exported["filtered_noise_csv"] = str(noise_csv)

        # 4. model_comparison.csv
        comp_csv = out_path / "model_comparison.csv"
        comp_data = report.get("model_comparison") or {}
        comp_rows = []
        for row in comp_data.get("polarity_models") or []:
            r_copy = dict(row)
            r_copy["task"] = "Polarity (3 lớp)"
            comp_rows.append(r_copy)
        for row in comp_data.get("emotion_models") or []:
            r_copy = dict(row)
            r_copy["task"] = "Emotion (7 cung bậc)"
            comp_rows.append(r_copy)
        df_comp = pd.DataFrame(comp_rows)
        has_gemma = _has_gemma_audit_rows(clean_rows)
        if not has_gemma and not df_comp.empty:
            drop_cols = [
                c
                for c in (
                    "agreement_vs_gemma_pct",
                    "macro_f1_vs_gemma",
                    "cohen_kappa_vs_gemma",
                    "score_5_mae_vs_gemma",
                )
                if c in df_comp.columns
            ]
            if drop_cols:
                df_comp = df_comp.drop(columns=drop_cols)
        df_comp.to_csv(comp_csv, index=False, encoding="utf-8-sig")
        exported["model_comparison_csv"] = str(comp_csv)

        return exported

    def run(
        self,
        tiktok_url: str | None = None,
        input_dir: str | Path | None = None,
        media_input: str | Path | Sequence[str | Path] | None = None,
        comments_source: str | Path | Sequence[str] | None = None,
        caption: str = "",
        enable_audio: bool = True,
        run_media_analysis: bool = True,
        text_column: str | None = None,
        unload_ollama_after_stage1: bool = True,
        run_ai_audit: bool = True,
        audit_limit: int | None = None,
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
        """Chạy pipeline 4 giai đoạn: dữ liệu thô vào data/<post_id>/, kết quả cuối ra outputs/<post_id>/."""
        if not tiktok_url and input_dir is None and media_input is None and comments_source is None:
            raise ValueError(
                "Cần cung cấp ít nhất `tiktok_url`, `input_dir`, `media_input`, hoặc `comments_source`."
            )

        data_root = Path(data_dir)
        effective_media = media_input
        effective_caption = caption
        effective_comments = comments_source

        if input_dir is not None:
            resolved_dir = resolve_local_input_dir(input_dir)
            if effective_comments is None:
                effective_comments = resolved_dir["comments_source"]
            if effective_media is None:
                effective_media = resolved_dir["media_input"]
            if not effective_caption.strip():
                effective_caption = resolved_dir["caption"]

        run_name = _infer_run_name(
            tiktok_url,
            effective_comments,
            effective_media,
            input_dir=input_dir,
        )

        out_path: Path | None = None
        if output_dir is not None:
            base_out = Path(output_dir)
            out_path = base_out / run_name if base_out.name in {"outputs", "output"} else base_out

        tiktok_bundle: TikTokMediaBundle | None = None

        # BƯỚC 0: Toàn bộ dữ liệu thu thập thô được lưu vào data/<post_id>/
        if tiktok_url:
            print(f"[Bước 0] Đang tải video/ảnh & metadata vào {data_root / run_name}...")
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

        # GIAI ĐOẠN 1 (AI ĐẦU VÀO)
        media_result: MediaAnalysisResult | None = None
        if effective_media is not None and run_media_analysis:
            print("[Giai đoạn 1] Gemma 4 E2B đang phân tích ngữ cảnh video/ảnh...")
            media_result = self.media_analyzer.analyze(
                media_input=effective_media,
                caption=effective_caption,
                enable_audio=enable_audio,
            )
        elif effective_media is not None and not run_media_analysis:
            print("[Giai đoạn 1] Bỏ qua phân tích media (run_media_analysis=False).")
        else:
            print("[Giai đoạn 1] Không có media đầu vào -> chạy thẳng trên file bình luận.")

        if unload_ollama_after_stage1 and media_result is not None:
            self.media_analyzer.unload_ollama_model()
            self.media_analyzer.unload_whisper_model()

        # GIAI ĐOẠN 2 & 3 (NLP + ML + DL SONG NGỮ VI/EN)
        preprocessed: list[PreprocessedComment] = []
        metadata_rows: list[dict[str, Any]] = []
        analyzed: list[AnalyzedComment] = []
        timings: dict[str, float] = {}
        summary: dict[str, Any] | None = None
        keywords_tfidf: dict[str, list[dict[str, Any]]] = {}

        if effective_comments is not None:
            df_comments = load_comments_dataframe(effective_comments, text_column=text_column)
            metadata_rows = df_comments.to_dict(orient="records")
            raw_texts = df_comments["comment"].tolist()

            print(f"[Giai đoạn 2] Lọc nhiễu, nhận diện ngôn ngữ (VI/EN) & tách từ ({len(raw_texts)} bình luận)...")
            preprocessed = self.text_preprocessor.process_batch(raw_texts)
            noise_summary = self._build_noise_summary(preprocessed)
            print(
                f"  -> Hợp lệ: {noise_summary['valid_comments']} | "
                f"Đã lọc nhiễu: {noise_summary['filtered_noise_count']}"
            )

            print("[Giai đoạn 3] Đang chạy ML cổ điển, Lexicon, Deep Learning (VI/EN) & Hybrid...")
            analyzed, timings = self.comment_analyzer.analyze_comments(
                preprocessed=preprocessed,
                metadata_rows=metadata_rows,
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
        if analyzed and run_ai_audit:
            print("[Giai đoạn 4] Gemma 4 E2B đang thẩm định thực tế bình luận theo ngữ cảnh...")
            t_aud = time.perf_counter()
            audits = self.ai_auditor.audit_comments(
                analyzed=analyzed,
                media_result=media_result,
                audit_limit=audit_limit,
                show_progress=show_progress,
            )
            timings["gemma_audit_sec"] = round(time.perf_counter() - t_aud, 3)
        elif analyzed and not run_ai_audit:
            print("[Giai đoạn 4] Đã bỏ qua thẩm định AI theo tùy chọn người dùng (run_ai_audit=False).")

        model_comparison = build_model_comparison(
            analyzed=analyzed,
            audits=audits,
            timings=timings,
        )

        if analyzed and run_ai_audit and audits:
            ai_verdict = self.ai_auditor.generate_overall_verdict(
                analyzed=analyzed,
                audits=audits,
                model_comparison=model_comparison.get("polarity_models") or [],
                media_result=media_result,
            )

        audit_map = {g.comment_index: g for g in audits}
        comments_payload = []
        for idx, item in enumerate(analyzed):
            row_dict = item.to_dict()
            g = audit_map.get(idx)
            row_dict["gemma_audit"] = g.to_dict() if g else None
            comments_payload.append(row_dict)

        noise_payload = [
            {
                "language": p.language,
                "original_text": p.original_text,
                "cleaned_text": p.cleaned_text,
                "filter_reason": p.filter_reason,
            }
            for p in preprocessed
            if not p.is_valid
        ]

        report: dict[str, Any] = {
            "run_name": run_name,
            "tiktok_metadata": tiktok_bundle.to_dict() if tiktok_bundle else None,
            "media_analysis": media_result.to_dict() if media_result else None,
            "noise_filter_summary": noise_summary,
            "keywords_tfidf": keywords_tfidf,
            "sentiment_summary": summary,
            "model_comparison": model_comparison,
            "ai_audit_verdict": ai_verdict,
            "comments": comments_payload,
            "filtered_noise": noise_payload,
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
        help="Thư mục cục bộ chứa sẵn file comment/media/metadata (vd: data/<post_id>).",
    )
    parser.add_argument("--media", nargs="+", default=None, help="Đường dẫn tới video/ảnh cục bộ.")
    parser.add_argument(
        "--comments",
        default=None,
        help="Đường dẫn tới file comment (.csv, .xlsx, .xls, .json, .txt).",
    )
    parser.add_argument("--caption", default="", help="Caption bài đăng TikTok (tùy chọn).")
    parser.add_argument("--text-column", default=None, help="Tên cột comment nếu dùng CSV/Excel/JSON.")
    parser.add_argument("--no-audio", action="store_true", help="Bỏ qua tách và nhận dạng giọng nói.")
    parser.add_argument(
        "--skip-media",
        action="store_true",
        help="Bỏ qua Giai đoạn 1 (không chạy Gemma 4 đọc video/ảnh).",
    )
    audit_group = parser.add_mutually_exclusive_group()
    audit_group.add_argument(
        "--ai-audit",
        dest="run_ai_audit",
        action="store_true",
        default=True,
        help="Bật Giai đoạn 4 (dùng Gemma 4 E2B thẩm định kết quả Giai đoạn 3 — mặc định).",
    )
    audit_group.add_argument(
        "--no-ai-audit",
        "--skip-audit",
        dest="run_ai_audit",
        action="store_false",
        help="Tắt Giai đoạn 4 (bỏ qua bước kiểm tra bằng Gemma 4 E2B).",
    )
    parser.add_argument("--audit-limit", type=int, default=None, help="Giới hạn số comment thẩm định.")
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

    media_input: str | list[str] | None = None
    if args.media:
        media_input = args.media[0] if len(args.media) == 1 else args.media

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
        device=args.device,
    )
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
        media_input=media_input,
        comments_source=args.comments,
        caption=args.caption,
        enable_audio=not args.no_audio,
        run_media_analysis=not args.skip_media,
        text_column=args.text_column,
        run_ai_audit=args.run_ai_audit,
        audit_limit=args.audit_limit,
        max_comments=args.max_comments,
        include_replies=not args.no_replies,
        max_replies_per_comment=args.max_replies_per_comment,
        crawl_headless=args.headless_crawl,
        crawl_interactive_pause=not args.no_crawl_pause,
        deduplicate_stats=args.deduplicate_stats,
        data_dir=args.data_dir,
        output_dir=args.output,
    )

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
        print("\n=== THỐNG KÊ ĐA TẦNG (SONG NGỮ VI/EN + THANG ĐIỂM KÉP) ===")
        print(f"- Ngôn ngữ     : {s['language_counts']}")
        print(f"- Polarity     : {s['polarity_counts']} | Tỷ lệ (%): {s['polarity_percentages']}")
        print(f"- Xu hướng     : {s['overall_tendency']}")
        print(
            f"- Điểm TB      : {s['avg_polarity_score']} [-1..+1] | "
            f"{s['avg_score_5']}/5 | {s['avg_score_10']}/10"
        )
        print(f"- 7 Cảm xúc (%): {s['emotion_percentages']}")

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

    if report.get("exported_files"):
        print("\n=== FILE ĐÃ XUẤT ===")
        for k, v in report["exported_files"].items():
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
