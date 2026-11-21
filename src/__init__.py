from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.comment_classifier import (
        DEFAULT_MODEL_NAME,
        LABEL_VI_MAP,
        CommentClassifier,
        CommentPrediction,
        preprocess_comment,
    )
    from src.media_analyzer import MediaAnalysisResult, MediaAnalyzer
    from src.pipeline import SentimentPipeline, load_comments, plot_sentiment_summary

__all__ = [
    "DEFAULT_MODEL_NAME",
    "LABEL_VI_MAP",
    "MediaAnalyzer",
    "MediaAnalysisResult",
    "CommentClassifier",
    "CommentPrediction",
    "SentimentPipeline",
    "load_comments",
    "plot_sentiment_summary",
    "preprocess_comment",
]

_EXPORT_MAP = {
    "DEFAULT_MODEL_NAME": ("src.comment_classifier", "DEFAULT_MODEL_NAME"),
    "LABEL_VI_MAP": ("src.comment_classifier", "LABEL_VI_MAP"),
    "CommentClassifier": ("src.comment_classifier", "CommentClassifier"),
    "CommentPrediction": ("src.comment_classifier", "CommentPrediction"),
    "preprocess_comment": ("src.comment_classifier", "preprocess_comment"),
    "MediaAnalyzer": ("src.media_analyzer", "MediaAnalyzer"),
    "MediaAnalysisResult": ("src.media_analyzer", "MediaAnalysisResult"),
    "SentimentPipeline": ("src.pipeline", "SentimentPipeline"),
    "load_comments": ("src.pipeline", "load_comments"),
    "plot_sentiment_summary": ("src.pipeline", "plot_sentiment_summary"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'src' has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)
