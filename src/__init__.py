from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.ingestion import (
        TikTokMediaBundle,
        collect_tiktok_post,
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
        AnalyzedComment,
        BilingualLexiconScorer,
        ClassicalMLSuite,
        CommentAnalyzer,
        GemmaAuditItem,
        GemmaAuditor,
        MediaAnalysisResult,
        MediaAnalyzer,
        build_model_comparison,
        compute_polarity_score,
    )
    from src.pipeline import (
        SentimentPipeline,
        build_detailed_model_outputs_df,
        format_final_results_df,
        resolve_local_input_dir,
    )
    from src.preprocessing import (
        PreprocessedComment,
        TextPreprocessor,
        clean_comment_text,
        detect_language,
        extract_keywords_tfidf,
    )
    from src.visualization import plot_sentiment_dashboard

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
    "SentimentPipeline",
    "TextPreprocessor",
    "TikTokMediaBundle",
    "build_detailed_model_outputs_df",
    "build_model_comparison",
    "clean_comment_text",
    "collect_tiktok_post",
    "compute_polarity_score",
    "detect_language",
    "extract_hashtags",
    "extract_keywords_tfidf",
    "extract_post_id",
    "format_final_results_df",
    "load_comments_dataframe",
    "plot_sentiment_dashboard",
    "resolve_local_input_dir",
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
    "SentimentPipeline": ("src.pipeline", "SentimentPipeline"),
    "TextPreprocessor": ("src.preprocessing.text_cleaner", "TextPreprocessor"),
    "TikTokMediaBundle": ("src.ingestion.tiktok_collector", "TikTokMediaBundle"),
    "build_detailed_model_outputs_df": ("src.pipeline", "build_detailed_model_outputs_df"),
    "build_model_comparison": ("src.models.gemma_auditor", "build_model_comparison"),
    "clean_comment_text": ("src.preprocessing.text_cleaner", "clean_comment_text"),
    "collect_tiktok_post": ("src.ingestion.tiktok_collector", "collect_tiktok_post"),
    "compute_polarity_score": ("src.models.sentiment_engine", "compute_polarity_score"),
    "detect_language": ("src.preprocessing.text_cleaner", "detect_language"),
    "extract_hashtags": ("src.ingestion.tiktok_collector", "extract_hashtags"),
    "extract_keywords_tfidf": ("src.preprocessing.text_cleaner", "extract_keywords_tfidf"),
    "extract_post_id": ("src.ingestion.tiktok_collector", "extract_post_id"),
    "format_final_results_df": ("src.pipeline", "format_final_results_df"),
    "load_comments_dataframe": ("src.ingestion.tiktok_collector", "load_comments_dataframe"),
    "plot_sentiment_dashboard": ("src.visualization", "plot_sentiment_dashboard"),
    "resolve_local_input_dir": ("src.pipeline", "resolve_local_input_dir"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'src' has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)
