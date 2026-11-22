from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.ingestion import (
        TikTokMediaBundle,
        crawl_tiktok_comments,
        download_tiktok_media,
        extract_hashtags,
        extract_post_id,
        load_comments_dataframe,
    )
    from src.models import (
        DEFAULT_EN_EMOTION_MODEL,
        DEFAULT_EN_SENTIMENT_MODEL,
        DEFAULT_PHOBERT_EMOTION_MODEL,
        DEFAULT_PHOBERT_SENTIMENT_MODEL,
        DEFAULT_VISOBERT_MODEL,
        EMOTION_LABELS,
        EMOTION_VI_MAP,
        LABEL_VI_MAP,
        RATING_5_VI_MAP,
        AnalyzedComment,
        BilingualLexiconScorer,
        ClassicalMLSuite,
        CommentAnalyzer,
        GemmaAuditItem,
        GemmaAuditor,
        MediaAnalysisResult,
        MediaAnalyzer,
        VietnameseLexiconScorer,
        build_model_comparison,
        compute_dual_scale,
    )
    from src.pipeline import SentimentPipeline, load_comments
    from src.preprocessing import (
        PreprocessedComment,
        TextPreprocessor,
        clean_comment_text,
        detect_language,
        extract_keywords_tfidf,
    )
    from src.visualization import plot_sentiment_dashboard, plot_sentiment_summary

__all__ = [
    "AnalyzedComment",
    "BilingualLexiconScorer",
    "ClassicalMLSuite",
    "CommentAnalyzer",
    "DEFAULT_EN_EMOTION_MODEL",
    "DEFAULT_EN_SENTIMENT_MODEL",
    "DEFAULT_PHOBERT_EMOTION_MODEL",
    "DEFAULT_PHOBERT_SENTIMENT_MODEL",
    "DEFAULT_VISOBERT_MODEL",
    "EMOTION_LABELS",
    "EMOTION_VI_MAP",
    "GemmaAuditItem",
    "GemmaAuditor",
    "LABEL_VI_MAP",
    "MediaAnalysisResult",
    "MediaAnalyzer",
    "PreprocessedComment",
    "RATING_5_VI_MAP",
    "SentimentPipeline",
    "TextPreprocessor",
    "TikTokMediaBundle",
    "VietnameseLexiconScorer",
    "build_model_comparison",
    "clean_comment_text",
    "compute_dual_scale",
    "crawl_tiktok_comments",
    "detect_language",
    "download_tiktok_media",
    "extract_hashtags",
    "extract_keywords_tfidf",
    "extract_post_id",
    "load_comments",
    "load_comments_dataframe",
    "plot_sentiment_dashboard",
    "plot_sentiment_summary",
]

_EXPORT_MAP = {
    "AnalyzedComment": ("src.models.sentiment_engine", "AnalyzedComment"),
    "BilingualLexiconScorer": ("src.models.classical_ml", "BilingualLexiconScorer"),
    "ClassicalMLSuite": ("src.models.classical_ml", "ClassicalMLSuite"),
    "CommentAnalyzer": ("src.models.sentiment_engine", "CommentAnalyzer"),
    "DEFAULT_EN_EMOTION_MODEL": ("src.models.sentiment_engine", "DEFAULT_EN_EMOTION_MODEL"),
    "DEFAULT_EN_SENTIMENT_MODEL": ("src.models.sentiment_engine", "DEFAULT_EN_SENTIMENT_MODEL"),
    "DEFAULT_PHOBERT_EMOTION_MODEL": ("src.models.sentiment_engine", "DEFAULT_PHOBERT_EMOTION_MODEL"),
    "DEFAULT_PHOBERT_SENTIMENT_MODEL": ("src.models.sentiment_engine", "DEFAULT_PHOBERT_SENTIMENT_MODEL"),
    "DEFAULT_VISOBERT_MODEL": ("src.models.sentiment_engine", "DEFAULT_VISOBERT_MODEL"),
    "EMOTION_LABELS": ("src.models.classical_ml", "EMOTION_LABELS"),
    "EMOTION_VI_MAP": ("src.models.classical_ml", "EMOTION_VI_MAP"),
    "GemmaAuditItem": ("src.models.gemma_auditor", "GemmaAuditItem"),
    "GemmaAuditor": ("src.models.gemma_auditor", "GemmaAuditor"),
    "LABEL_VI_MAP": ("src.models.sentiment_engine", "LABEL_VI_MAP"),
    "MediaAnalysisResult": ("src.models.media_context", "MediaAnalysisResult"),
    "MediaAnalyzer": ("src.models.media_context", "MediaAnalyzer"),
    "PreprocessedComment": ("src.preprocessing.text_cleaner", "PreprocessedComment"),
    "RATING_5_VI_MAP": ("src.models.sentiment_engine", "RATING_5_VI_MAP"),
    "SentimentPipeline": ("src.pipeline", "SentimentPipeline"),
    "TextPreprocessor": ("src.preprocessing.text_cleaner", "TextPreprocessor"),
    "TikTokMediaBundle": ("src.ingestion.tiktok_collector", "TikTokMediaBundle"),
    "VietnameseLexiconScorer": ("src.models.classical_ml", "VietnameseLexiconScorer"),
    "build_model_comparison": ("src.models.gemma_auditor", "build_model_comparison"),
    "clean_comment_text": ("src.preprocessing.text_cleaner", "clean_comment_text"),
    "compute_dual_scale": ("src.models.sentiment_engine", "compute_dual_scale"),
    "crawl_tiktok_comments": ("src.ingestion.tiktok_collector", "crawl_tiktok_comments"),
    "detect_language": ("src.preprocessing.text_cleaner", "detect_language"),
    "download_tiktok_media": ("src.ingestion.tiktok_collector", "download_tiktok_media"),
    "extract_hashtags": ("src.ingestion.tiktok_collector", "extract_hashtags"),
    "extract_keywords_tfidf": ("src.preprocessing.text_cleaner", "extract_keywords_tfidf"),
    "extract_post_id": ("src.ingestion.tiktok_collector", "extract_post_id"),
    "load_comments": ("src.pipeline", "load_comments"),
    "load_comments_dataframe": ("src.ingestion.tiktok_collector", "load_comments_dataframe"),
    "plot_sentiment_dashboard": ("src.visualization", "plot_sentiment_dashboard"),
    "plot_sentiment_summary": ("src.visualization", "plot_sentiment_summary"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'src' has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)
