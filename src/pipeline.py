from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from src.comment_classifier import (
    DEFAULT_MODEL_NAME,
    CommentClassifier,
    CommentPrediction,
)
from src.media_analyzer import MediaAnalysisResult, MediaAnalyzer

COMMENT_COLUMN_CANDIDATES = (
    "comment",
    "text",
    "content",
    "noi_dung",
    "binh_luan",
    "message",
)


def load_comments(
    comments_source: str | Path | Sequence[str],
    text_column: str | None = None,
) -> list[str]:
    """Đọc danh sách bình luận từ list Python hoặc file CSV / JSON / TXT."""
    if not isinstance(comments_source, (str, Path)):
        return [str(item).strip() for item in comments_source if str(item).strip()]

    path = Path(comments_source)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file comment: {path}")

    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
        if df.empty:
            return []

        col = text_column
        if col is None:
            lower_map = {c.lower(): c for c in df.columns}
            for candidate in COMMENT_COLUMN_CANDIDATES:
                if candidate in lower_map:
                    col = lower_map[candidate]
                    break
            if col is None:
                col = df.columns[0]
        elif col not in df.columns:
            raise ValueError(
                f"Không tìm thấy cột `{col}` trong {path}. Các cột hiện có: {list(df.columns)}"
            )

        return [
            str(val).strip()
            for val in df[col].dropna().tolist()
            if str(val).strip()
        ]

    if ext == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            if "comments" in data and isinstance(data["comments"], list):
                data = data["comments"]
            else:
                raise ValueError("File JSON dạng object cần có key `comments` chứa danh sách.")

        if not isinstance(data, list):
            raise ValueError("File JSON comment phải là danh sách chuỗi hoặc object.")

        comments: list[str] = []
        for item in data:
            if isinstance(item, str):
                if item.strip():
                    comments.append(item.strip())
            elif isinstance(item, dict):
                key = text_column
                if key is None:
                    lower_map = {k.lower(): k for k in item.keys()}
                    for candidate in COMMENT_COLUMN_CANDIDATES:
                        if candidate in lower_map:
                            key = lower_map[candidate]
                            break
                    if key is None and item:
                        key = next(iter(item.keys()))
                if key and item.get(key) is not None and str(item[key]).strip():
                    comments.append(str(item[key]).strip())
        return comments

    if ext == ".txt":
        with path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    raise ValueError(f"Định dạng file comment chưa hỗ trợ: {path.suffix} (chỉ hỗ trợ .csv, .json, .txt)")


class SentimentPipeline:
    """Pipeline 2 giai đoạn: Gemma 4 E2B phân tích media + ViSoBERT phân loại comment."""

    def __init__(
        self,
        media_analyzer: MediaAnalyzer | None = None,
        comment_classifier: CommentClassifier | None = None,
    ) -> None:
        self.media_analyzer = media_analyzer or MediaAnalyzer()
        self.comment_classifier = comment_classifier or CommentClassifier()

    def run(
        self,
        media_input: str | Path | Sequence[str | Path] | None = None,
        comments_source: str | Path | Sequence[str] | None = None,
        caption: str = "",
        enable_audio: bool = True,
        text_column: str | None = None,
        unload_ollama_after_stage1: bool = False,
        output_dir: str | Path | None = None,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Chạy tuần tự Giai đoạn 1 và Giai đoạn 2, tùy chọn xuất báo cáo ra đĩa."""
        if media_input is None and comments_source is None:
            raise ValueError("Cần cung cấp ít nhất `media_input` hoặc `comments_source`.")

        media_result: MediaAnalysisResult | None = None
        if media_input is not None:
            media_result = self.media_analyzer.analyze(
                media_input=media_input,
                caption=caption,
                enable_audio=enable_audio,
            )

        if unload_ollama_after_stage1 or self.comment_classifier.device == "cuda":
            self.media_analyzer.unload_ollama_model()

        predictions: list[CommentPrediction] = []
        summary: dict[str, Any] | None = None
        if comments_source is not None:
            comments = load_comments(comments_source, text_column=text_column)
            predictions = self.comment_classifier.predict_batch(
                comments, show_progress=show_progress
            )
            summary = self.comment_classifier.summarize(predictions)

        report: dict[str, Any] = {
            "media_analysis": media_result.to_dict() if media_result else None,
            "sentiment_summary": summary,
            "comments": [pred.to_dict() for pred in predictions],
        }

        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            report_file = out_path / "report.json"
            with report_file.open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            if predictions:
                rows = [
                    {
                        "original_text": p.original_text,
                        "cleaned_text": p.cleaned_text,
                        "label": p.label,
                        "label_vi": p.label_vi,
                        "confidence": p.confidence,
                        "score_POS": p.scores.get("POS", 0.0),
                        "score_NEG": p.scores.get("NEG", 0.0),
                        "score_NEU": p.scores.get("NEU", 0.0),
                    }
                    for p in predictions
                ]
                pd.DataFrame(rows).to_csv(
                    out_path / "classified_comments.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

        return report


def plot_sentiment_summary(
    report: dict[str, Any],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Vẽ biểu đồ cột ngang và biểu đồ tròn phân bố cảm xúc bình luận."""
    summary = report.get("sentiment_summary")
    if not summary or summary.get("total_comments", 0) == 0:
        raise ValueError("Báo cáo chưa có dữ liệu thống kê cảm xúc để vẽ biểu đồ.")

    counts = summary["counts"]
    labels = ["Tích cực", "Trung tính", "Tiêu cực"]
    values = [counts.get("POS", 0), counts.get("NEU", 0), counts.get("NEG", 0)]
    colors = ["#2ecc71", "#95a5a6", "#e74c3c"]

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(12, 4.8))

    bars = ax_bar.barh(labels, values, color=colors)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Số lượng bình luận")
    ax_bar.set_title("Số lượng theo nhãn cảm xúc")
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, values):
        ax_bar.text(
            bar.get_width() + max(values, default=1) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
        )

    non_zero = [(lbl, val, col) for lbl, val, col in zip(labels, values, colors) if val > 0]
    if non_zero:
        ax_pie.pie(
            [v for _, v, _ in non_zero],
            labels=[l for l, _, _ in non_zero],
            colors=[c for _, _, c in non_zero],
            autopct="%1.1f%%",
            startangle=90,
        )
    ax_pie.set_title(f"Tỷ lệ cảm xúc (Xu hướng: {summary.get('overall_tendency', '')})")

    media_info = report.get("media_analysis") or {}
    topic = media_info.get("chu_de_chinh")
    if topic:
        fig.suptitle(f"Chủ đề: {topic}", fontsize=13, fontweight="bold")

    fig.tight_layout()
    if save_path is not None:
        out_img = Path(save_path)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_img, dpi=150, bbox_inches="tight")

    return fig


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hệ thống phân tích nội dung TikTok (Gemma 4 E2B) & cảm xúc comment (ViSoBERT)."
    )
    parser.add_argument(
        "--media",
        nargs="+",
        default=None,
        help="Đường dẫn tới video, ảnh hoặc danh sách ảnh / thư mục ảnh.",
    )
    parser.add_argument(
        "--comments",
        default=None,
        help="Đường dẫn tới file comment (.csv, .json, .txt).",
    )
    parser.add_argument(
        "--caption",
        default="",
        help="Caption hoặc mô tả kèm theo bài đăng TikTok (tùy chọn).",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Tên cột chứa nội dung comment nếu dùng CSV/JSON.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Bỏ qua bước tách và nhận dạng giọng nói trong video.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Thiết bị chạy mô hình ViSoBERT (mặc định: cpu để tiết kiệm VRAM).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="URL của Ollama API.",
    )
    parser.add_argument(
        "--ollama-model",
        default="gemma4-e2b-opt",
        help="Tên model Ollama dùng cho Giai đoạn 1.",
    )
    parser.add_argument(
        "--visobert-model",
        default=DEFAULT_MODEL_NAME,
        help="Model Hugging Face dùng cho Giai đoạn 2.",
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        help="Kích thước model faster-whisper (tiny, base, small...).",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Thư mục lưu report.json và classified_comments.csv.",
    )
    parser.add_argument(
        "--save-chart",
        default=None,
        help="Đường dẫn file ảnh để lưu biểu đồ cảm xúc (tùy chọn).",
    )
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
    classifier = CommentClassifier(
        model_name=args.visobert_model,
        device=args.device,
    )
    pipeline = SentimentPipeline(
        media_analyzer=analyzer,
        comment_classifier=classifier,
    )

    report = pipeline.run(
        media_input=media_input,
        comments_source=args.comments,
        caption=args.caption,
        enable_audio=not args.no_audio,
        text_column=args.text_column,
        output_dir=args.output,
    )

    print("\n=== KẾT QUẢ PHÂN TÍCH ===")
    if report["media_analysis"]:
        m = report["media_analysis"]
        print(f"- Chủ đề chính : {m['chu_de_chinh']}")
        print(f"- Quan điểm    : {m['quan_diem_noi_dung']}")
        print(f"- Tóm tắt      : {m['tom_tat_ngan']}")
        print(f"- Từ khóa      : {', '.join(m['tu_khoa'])}")
        print(f"- Âm thanh     : {'Có' if m['has_audio'] else 'Không'}")
        if m["transcript"]:
            print(f"- Lời thoại    : {m['transcript'][:200]}...")

    if report["sentiment_summary"]:
        s = report["sentiment_summary"]
        print("\n=== THỐNG KÊ BÌNH LUẬN (ViSoBERT) ===")
        print(f"- Tổng comment : {s['total_comments']}")
        print(f"- Phân bố      : {s['counts']} | Tỷ lệ (%): {s['percentages']}")
        print(f"- Xu hướng     : {s['overall_tendency']}")

        if args.save_chart:
            fig = plot_sentiment_summary(report, save_path=args.save_chart)
            plt.close(fig)
            print(f"- Đã lưu biểu đồ tại: {args.save_chart}")

    print(f"\nĐã lưu báo cáo chi tiết vào thư mục: {args.output}")


if __name__ == "__main__":
    main()
