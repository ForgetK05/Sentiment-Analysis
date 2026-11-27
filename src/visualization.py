from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.models.classical_ml import EMOTION_LABELS, EMOTION_VI_MAP, POLARITY_MODEL_META

# Đảm bảo font tiếng Việt hiển thị sạch trên biểu đồ Windows
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False

_FAMILY_SHORT_TAG: dict[str, str] = {
    "Classical ML (Generative)": "[Generative]",
    "Classical ML (Linear)": "[Linear]",
    "Classical ML (Margin / SVM)": "[SVM]",
    "Classical ML (Tree Bagging)": "[Tree-Bag]",
    "Classical ML (Tree Boosting)": "[Tree-Boost]",
    "Classical ML Ensemble": "[Ensemble]",
    "Rule-based NLP": "[Rule]",
    "Deep Learning": "[DL]",
    "Hybrid ML + DL": "[Hybrid]",
    "Multimodal LLM Reference": "[LLM Ref]",
}

_FAMILY_COLORS: dict[str, str] = {
    "Classical ML (Generative)": "#0ea5e9",
    "Classical ML (Linear)": "#6366f1",
    "Classical ML (Margin / SVM)": "#8b5cf6",
    "Classical ML (Tree Bagging)": "#14b8a6",
    "Classical ML (Tree Boosting)": "#f59e0b",
    "Classical ML Ensemble": "#2563eb",
    "Rule-based NLP": "#64748b",
    "Deep Learning": "#ec4899",
    "Hybrid ML + DL": "#16a34a",
}


def _short_model_label(row: dict[str, Any]) -> str:
    name = str(row.get("model", "")).replace(" (Hệ thống chính)", " (Chính)")
    name = name.replace("TF-IDF + ", "").replace("SVD+Social + ", "").replace("SVD + ", "")
    name = name.replace(" (NB-Weighted SVM)", "").replace("HistGradientBoosting", "HistGB")
    tag = _FAMILY_SHORT_TAG.get(str(row.get("family", "")), "")
    return f"{tag} {name}".strip() if tag else name


def _clean_feature_name(raw_feat: str) -> str:
    s = str(raw_feat or "")
    if s.startswith("word_tfidf__"):
        return f"từ: {s[len('word_tfidf__'):]}"
    if s.startswith("char_tfidf__"):
        return f"ký tự: {s[len('char_tfidf__'):].strip() or s[len('char_tfidf__'):]}"
    if s.startswith("social_lex__"):
        return f"tín hiệu: {s[len('social_lex__'):]}"
    return s


def _dominant_language(summary: dict[str, Any], benchmark: dict[str, Any]) -> str:
    lang_counts = summary.get("language_counts") or {}
    vi_cnt = int(lang_counts.get("vi", 0) or 0)
    en_cnt = int(lang_counts.get("en", 0) or 0)
    preferred = "vi" if vi_cnt >= en_cnt else "en"
    if preferred in benchmark:
        return preferred
    for cand in ("vi", "en"):
        if cand in benchmark:
            return cand
    return preferred


def _build_suptitle(report: dict[str, Any], title_prefix: str = "") -> str:
    summary = report.get("sentiment_summary") or {}
    media_info = report.get("media_analysis") or {}
    topic = media_info.get("chu_de_chinh")
    noise_info = report.get("noise_filter_summary") or {}
    lang_counts = summary.get("language_counts") or {}
    lang_str = f"VI: {lang_counts.get('vi', 0)} | EN: {lang_counts.get('en', 0)}"
    engine = "Classical ML" if summary.get("engine_mode") == "classical" else "Hybrid ML + DL"
    subtitle = (
        f"Chế độ: {engine} | Hợp lệ: {noise_info.get('valid_comments', summary.get('total_valid_comments', 0))} "
        f"({lang_str}) | Đã lọc nhiễu: {noise_info.get('filtered_noise_count', 0)}"
    )
    if topic:
        head = f"{title_prefix}Chủ đề: {topic}" if title_prefix else f"Chủ đề: {topic}"
        return f"{head}\n({subtitle})"
    head = title_prefix.rstrip(" —-") if title_prefix else "Dashboard Phân tích Cảm xúc"
    return f"{head}\n({subtitle})"


def _draw_polarity_panel(ax: plt.Axes, report: dict[str, Any]) -> None:
    summary = report.get("sentiment_summary") or {}
    pol_counts = summary.get("polarity_counts") or {}
    total_valid = max(1, int(summary.get("total_valid_comments", 0) or sum(pol_counts.values()) or 1))
    pol_labels = ["Tích cực (POS)", "Trung tính (NEU)", "Tiêu cực (NEG)"]
    sys_vals = [int(pol_counts.get("POS", 0)), int(pol_counts.get("NEU", 0)), int(pol_counts.get("NEG", 0))]

    comp = report.get("model_comparison") or {}
    pol_models = comp.get("polarity_models") or []
    gemma_row = next((r for r in pol_models if "Gemma 4" in str(r.get("model", ""))), None)

    x_idx = [0, 1, 2]
    if gemma_row:
        g_vals = [int(gemma_row["pos_count"]), int(gemma_row["neu_count"]), int(gemma_row["neg_count"])]
        w = 0.36
        b1 = ax.bar([x - w / 2 for x in x_idx], sys_vals, width=w, label="Hệ thống", color="#3498db")
        b2 = ax.bar([x + w / 2 for x in x_idx], g_vals, width=w, label="Gemma 4 E2B", color="#e67e22")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False, fontsize=8.5)
        max_v = max(sys_vals + g_vals + [1])
        for bar in list(b1) + list(b2):
            h_v = int(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h_v + max_v * 0.02,
                f"{h_v}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        ax.set_ylim(0, max_v * 1.2 + 2)
        ax.set_title("Sắc thái Polarity: Hệ thống vs Gemma 4 E2B", pad=22)
    else:
        colors = ["#2ecc71", "#95a5a6", "#e74c3c"]
        bars = ax.bar(x_idx, sys_vals, color=colors)
        max_v = max(sys_vals + [1])
        for bar in bars:
            h_v = int(bar.get_height())
            pct = h_v * 100.0 / total_valid
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h_v + max_v * 0.02,
                f"{h_v}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        ax.set_ylim(0, max_v * 1.24 + 2)
        ax.set_title(f"Phân bố Sắc thái ({summary.get('overall_tendency', '')})")

    ax.set_xticks(x_idx)
    ax.set_xticklabels(pol_labels)
    ax.set_ylabel("Số bình luận")
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def _draw_emotion_panel(ax: plt.Axes, summary: dict[str, Any]) -> None:
    emo_counts = summary.get("emotion_counts") or {}
    total_valid = max(1, int(summary.get("total_valid_comments", 0) or sum(emo_counts.values()) or 1))
    emo_items = [(EMOTION_VI_MAP[e], int(emo_counts.get(e, 0))) for e in EMOTION_LABELS]
    emo_items.sort(key=lambda x: x[1], reverse=True)
    emo_names = [k for k, _ in emo_items]
    emo_vals = [v for _, v in emo_items]
    emo_colors = ["#e74c3c", "#9b59b6", "#34495e", "#2ecc71", "#f39c12", "#1abc9c", "#95a5a6"]
    bars_e = ax.barh(emo_names, emo_vals, color=emo_colors[: len(emo_names)])
    ax.invert_yaxis()
    ax.set_xlabel("Số bình luận")
    ax.set_title("Phân bố 7 cung bậc cảm xúc")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    max_e = max(emo_vals, default=1) or 1
    ax.set_xlim(0, max_e * 1.28 + 2)
    for bar, val in zip(bars_e, emo_vals):
        pct = val * 100.0 / total_valid
        label = f"{val} ({pct:.1f}%)" if val > 0 else "0"
        ax.text(
            bar.get_width() + max_e * 0.02,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=8.5,
        )


def _draw_context_panel(ax: plt.Axes, summary: dict[str, Any]) -> None:
    pol_counts = summary.get("polarity_counts") or {}
    on_topic_info = summary.get("on_topic_summary")
    if on_topic_info:
        on_pol = on_topic_info.get("on_topic_polarity_counts") or {}
        on_vals = [int(on_pol.get("POS", 0)), int(on_pol.get("NEU", 0)), int(on_pol.get("NEG", 0))]
        off_vals = [
            max(0, int(pol_counts.get("POS", 0)) - on_vals[0]),
            max(0, int(pol_counts.get("NEU", 0)) - on_vals[1]),
            max(0, int(pol_counts.get("NEG", 0)) - on_vals[2]),
        ]
        ctx_x = [0, 1, 2]
        w_ctx = 0.36
        b_on = ax.bar(
            [x - w_ctx / 2 for x in ctx_x],
            on_vals,
            width=w_ctx,
            label=f"Bám sát video ({on_topic_info.get('on_topic_count', 0)})",
            color="#2980b9",
        )
        b_off = ax.bar(
            [x + w_ctx / 2 for x in ctx_x],
            off_vals,
            width=w_ctx,
            label=f"Ngoài lề ({on_topic_info.get('off_topic_count', 0)})",
            color="#bdc3c7",
        )
        max_v = max(on_vals + off_vals + [1])
        for bar in list(b_on) + list(b_off):
            h_v = int(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h_v + max_v * 0.02,
                str(h_v),
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        ax.set_ylim(0, max_v * 1.22 + 2)
        ax.set_xticks(ctx_x)
        ax.set_xticklabels(["Tích cực (POS)", "Trung tính (NEU)", "Tiêu cực (NEG)"])
        ax.set_ylabel("Số bình luận")
        ax.set_title(
            f"Bám sát nội dung video: {on_topic_info.get('on_topic_pct', 0)}% "
            f"({on_topic_info.get('on_topic_count', 0)}/{summary.get('total_valid_comments', 0)})"
        )
        ax.legend(fontsize=8.5, frameon=False)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    else:
        lang_map = summary.get("language_counts") or {}
        lang_labels = ["Tiếng Việt (vi)", "Tiếng Anh (en)"]
        lang_vals = [int(lang_map.get("vi", 0)), int(lang_map.get("en", 0))]
        max_l = max(lang_vals + [1])
        bars_l = ax.bar(lang_labels, lang_vals, color=["#1abc9c", "#3498db"])
        for bar in bars_l:
            h_v = int(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h_v + max_l * 0.02,
                str(h_v),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_ylim(0, max_l * 1.22 + 2)
        ax.set_ylabel("Số bình luận")
        ax.set_title(
            f"Phân bố ngôn ngữ (Cực tính TB: {summary.get('avg_polarity_score', 0)} | "
            f"Tự tin TB: {summary.get('avg_confidence', 0)})"
        )
        ax.grid(axis="y", linestyle="--", alpha=0.35)


def _draw_keywords_panel(ax: plt.Axes, report: dict[str, Any]) -> None:
    kw_data = report.get("keywords_tfidf") or {}
    pos_kw = list(kw_data.get("positive") or [])[:6]
    neg_kw = list(kw_data.get("negative") or [])[:6]
    overall_kw = list(kw_data.get("overall") or [])[:10]

    if pos_kw or neg_kw:
        labels: list[str] = []
        vals: list[float] = []
        colors: list[str] = []
        for item in pos_kw:
            labels.append(f"+ {item['keyword']}")
            vals.append(float(item.get("tfidf_score", 0.0)))
            colors.append("#2ecc71")
        for item in neg_kw:
            labels.append(f"- {item['keyword']}")
            vals.append(-float(item.get("tfidf_score", 0.0)))
            colors.append("#e74c3c")

        y_pos = list(range(len(labels)))
        bars = ax.barh(y_pos, vals, color=colors)
        ax.axvline(0, color="#64748b", linewidth=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.invert_yaxis()
        max_abs = max((abs(v) for v in vals), default=0.05) or 0.05
        ax.set_xlim(-max_abs * 1.30, max_abs * 1.30)
        for bar, v in zip(bars, vals):
            offset = max_abs * 0.03
            ha = "left" if v >= 0 else "right"
            x_txt = v + offset if v >= 0 else v - offset
            ax.text(x_txt, bar.get_y() + bar.get_height() / 2, f"{abs(v):.3f}", va="center", ha=ha, fontsize=8)
        ax.set_xlabel("Điểm TF-IDF (Trái: Tiêu cực | Phải: Tích cực)")
        ax.set_title("Từ khóa nổi bật theo sắc thái (TF-IDF)")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
    elif overall_kw:
        labels = [str(item["keyword"]) for item in overall_kw]
        vals = [float(item.get("tfidf_score", 0.0)) for item in overall_kw]
        bars = ax.barh(labels, vals, color="#3b82f6")
        ax.invert_yaxis()
        max_v = max(vals, default=0.05) or 0.05
        ax.set_xlim(0, max_v * 1.25)
        for bar, v in zip(bars, vals):
            ax.text(v + max_v * 0.02, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlabel("Điểm TF-IDF")
        ax.set_title("Top từ khóa nổi bật (TF-IDF)")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
    else:
        ax.text(0.5, 0.5, "Chưa đủ từ khóa TF-IDF", ha="center", va="center", fontsize=10, color="#64748b")
        ax.set_axis_off()


def _draw_model_comparison_panel(ax: plt.Axes, report: dict[str, Any]) -> None:
    comp = report.get("model_comparison") or {}
    pol_models = comp.get("polarity_models") or []
    non_ref_models = [r for r in pol_models if "Gemma 4" not in str(r.get("model", ""))]
    has_dl_models = any(r.get("family") == "Deep Learning" for r in non_ref_models)

    if not non_ref_models:
        ax.text(0.5, 0.5, "Không có dữ liệu so sánh mô hình", ha="center", va="center", fontsize=10, color="#64748b")
        ax.set_axis_off()
        return

    m_names = [_short_model_label(r) for r in non_ref_models]
    y_pos = list(range(len(m_names)))
    has_audit_scores = non_ref_models[0].get("agreement_vs_gemma_pct") is not None

    # Tìm ranh giới giữa tầng cơ sở (base) và tầng tổ hợp (ensemble / system)
    split_idx: int | None = None
    for idx, r in enumerate(non_ref_models):
        tier = r.get("tier")
        fam = str(r.get("family", ""))
        if tier in {"ensemble", "system"} or "Ensemble" in fam or "Hybrid" in fam:
            split_idx = idx
            break

    if has_audit_scores:
        agrees = [float(r.get("agreement_vs_gemma_pct") or 0.0) for r in non_ref_models]
        f1s = [float(r.get("macro_f1_vs_gemma") or 0.0) * 100.0 for r in non_ref_models]
        h = 0.36
        ax.barh([y - h / 2 for y in y_pos], agrees, height=h, label="Agreement (%) vs Gemma 4", color="#2980b9")
        ax.barh([y + h / 2 for y in y_pos], f1s, height=h, label="Macro-F1 (%) vs Gemma 4", color="#8e44ad")
        ax.set_xlim(0, 112)
        ax.set_xlabel("Tỷ lệ khớp thực tế (%)")
        ax.set_title(
            "Hiệu năng theo Họ mô hình vs Gemma 4" if has_dl_models else "Hiệu năng các họ Classical ML vs Gemma 4",
            pad=28,
        )
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=8.5)
    else:
        pos_p = [float(r["pos_pct"]) for r in non_ref_models]
        neu_p = [float(r["neu_pct"]) for r in non_ref_models]
        neg_p = [float(r["neg_pct"]) for r in non_ref_models]
        ax.barh(y_pos, pos_p, label="POS %", color="#2ecc71")
        ax.barh(y_pos, neu_p, left=pos_p, label="NEU %", color="#95a5a6")
        left_neg = [a + b for a, b in zip(pos_p, neu_p)]
        ax.barh(y_pos, neg_p, left=left_neg, label="NEG %", color="#e74c3c")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Phân bố (%)")
        ax.set_title(
            "Phân bố dự đoán theo Họ mô hình (ML & DL)"
            if has_dl_models
            else "Phân bố dự đoán theo 5 Họ Classical ML",
            pad=28,
        )
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=8.5)

    if split_idx is not None and split_idx > 0:
        ax.axhline(split_idx - 0.5, color="#0f172a", linestyle="--", linewidth=1.1, alpha=0.75)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(m_names, fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.25)


def plot_sentiment_dashboard(
    report: dict[str, Any],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Dashboard 2x2 tổng hợp (tương thích ngược, đã sửa lỗi cắt chữ và legend che cột)."""
    summary = report.get("sentiment_summary")
    if not summary or summary.get("total_valid_comments", 0) == 0:
        raise ValueError("Báo cáo chưa có dữ liệu thống kê bình luận để vẽ biểu đồ.")

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    _draw_polarity_panel(axes[0, 0], report)
    _draw_emotion_panel(axes[0, 1], summary)
    _draw_context_panel(axes[1, 0], summary)
    _draw_model_comparison_panel(axes[1, 1], report)

    fig.suptitle(_build_suptitle(report), fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.subplots_adjust(hspace=0.36, wspace=0.28)
    if save_path is not None:
        out_img = Path(save_path)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_img, dpi=150, bbox_inches="tight")

    return fig


def plot_overview_dashboard(
    report: dict[str, Any],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Dashboard Tổng quan Nội dung & Khán giả (2x2): Polarity, 7 Cảm xúc, Bám sát video, Top từ khóa."""
    summary = report.get("sentiment_summary")
    if not summary or summary.get("total_valid_comments", 0) == 0:
        raise ValueError("Báo cáo chưa có dữ liệu thống kê bình luận để vẽ biểu đồ.")

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0))
    _draw_polarity_panel(axes[0, 0], report)
    _draw_emotion_panel(axes[0, 1], summary)
    _draw_context_panel(axes[1, 0], summary)
    _draw_keywords_panel(axes[1, 1], report)

    fig.suptitle(_build_suptitle(report, "Tổng quan Khán giả — "), fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.subplots_adjust(hspace=0.34, wspace=0.28)
    if save_path is not None:
        out_img = Path(save_path)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_img, dpi=150, bbox_inches="tight")

    return fig


def plot_models_dashboard(
    report: dict[str, Any],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Dashboard Kỹ thuật Mô hình (2x2): Dự đoán theo Họ, Benchmark Test Macro-F1, Tốc độ & Top Features."""
    summary = report.get("sentiment_summary")
    if not summary or summary.get("total_valid_comments", 0) == 0:
        raise ValueError("Báo cáo chưa có dữ liệu thống kê bình luận để vẽ biểu đồ.")

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.8))
    ax_pred, ax_bench = axes[0, 0], axes[0, 1]
    ax_speed, ax_feat = axes[1, 0], axes[1, 1]

    # [0,0] Phân bố dự đoán hoặc Agreement vs Gemma
    _draw_model_comparison_panel(ax_pred, report)

    # [0,1] Benchmark Test Macro-F1 cho ngôn ngữ chính của video
    benchmark = summary.get("classical_ml_benchmark") or {}
    dom_lang = _dominant_language(summary, benchmark)
    lang_bench = benchmark.get(dom_lang) or {}
    pol_bench = lang_bench.get("polarity_models") or {}

    if pol_bench:
        items_b = sorted(
            pol_bench.items(),
            key=lambda kv: float((kv[1] or {}).get("macro_f1") or 0.0),
            reverse=True,
        )
        b_labels = []
        b_scores = []
        b_colors = []
        b_edges = []
        for m_key, m_val in items_b:
            meta = POLARITY_MODEL_META.get(m_key, {})
            short = meta.get("short_name", m_key).replace("HistGradientBoosting", "HistGB")
            tag = _FAMILY_SHORT_TAG.get(meta.get("family", ""), "")
            b_labels.append(f"{tag} {short}".strip())
            b_scores.append(float((m_val or {}).get("macro_f1") or 0.0) * 100.0)
            b_colors.append(meta.get("family_color", "#64748b"))
            b_edges.append("#0f172a" if m_key == "ml_ensemble" else "none")

        y_b = list(range(len(b_labels)))
        bars_b = ax_bench.barh(y_b, b_scores, color=b_colors, edgecolor=b_edges, linewidth=1.5)
        ax_bench.set_yticks(y_b)
        ax_bench.set_yticklabels(b_labels, fontsize=8)
        ax_bench.invert_yaxis()
        max_b = max(b_scores, default=80.0) or 80.0
        ax_bench.set_xlim(0, min(105.0, max_b * 1.18))
        for bar, sc in zip(bars_b, b_scores):
            ax_bench.text(sc + 1.0, bar.get_y() + bar.get_height() / 2, f"{sc:.1f}%", va="center", fontsize=8)
        ax_bench.set_xlabel("Macro-F1 trên tập Test chuẩn (%)")
        ax_bench.set_title(f"Benchmark Tập Test Chuẩn ({dom_lang.upper()}) — 5 Họ ML & Ensemble")
        ax_bench.grid(axis="x", linestyle="--", alpha=0.35)
    else:
        ax_bench.text(0.5, 0.5, "Chưa có dữ liệu Benchmark", ha="center", va="center", color="#64748b")
        ax_bench.set_axis_off()

    # [1,0] Tốc độ suy luận riêng từng mô hình (ms)
    comp = report.get("model_comparison") or {}
    pol_models = comp.get("polarity_models") or []
    speed_rows = [
        r for r in pol_models
        if r.get("inference_time_sec") is not None and "Gemma 4" not in str(r.get("model", ""))
    ]
    if speed_rows:
        s_labels = [_short_model_label(r) for r in speed_rows]
        s_ms = [float(r["inference_time_sec"]) * 1000.0 for r in speed_rows]
        s_colors = [_FAMILY_COLORS.get(str(r.get("family", "")), "#64748b") for r in speed_rows]
        y_s = list(range(len(s_labels)))
        bars_s = ax_speed.barh(y_s, s_ms, color=s_colors)
        ax_speed.set_yticks(y_s)
        ax_speed.set_yticklabels(s_labels, fontsize=8)
        ax_speed.invert_yaxis()
        max_ms = max(s_ms, default=10.0) or 10.0
        ax_speed.set_xlim(0, max_ms * 1.25 + 1.0)
        for bar, ms_val in zip(bars_s, s_ms):
            txt = f"{ms_val:.1f} ms" if ms_val < 1000.0 else f"{ms_val / 1000.0:.2f} s"
            ax_speed.text(ms_val + max_ms * 0.02, bar.get_y() + bar.get_height() / 2, txt, va="center", fontsize=8)
        ax_speed.set_xlabel("Thời gian suy luận (ms)")
        ax_speed.set_title("Tốc độ suy luận từng mô hình trên video này")
        ax_speed.grid(axis="x", linestyle="--", alpha=0.35)
    else:
        ax_speed.text(0.5, 0.5, "Chưa có dữ liệu thời gian suy luận", ha="center", va="center", color="#64748b")
        ax_speed.set_axis_off()

    # [1,1] Giải thích mô hình: Top Features POS vs NEG
    top_pol_feats = (lang_bench.get("top_features") or {}).get("polarity") or {}
    pos_feats = list(top_pol_feats.get("POS") or [])[:6]
    neg_feats = list(top_pol_feats.get("NEG") or [])[:6]
    if pos_feats or neg_feats:
        f_labels: list[str] = []
        f_weights: list[float] = []
        f_colors: list[str] = []
        for item in pos_feats:
            f_labels.append(_clean_feature_name(item.get("feature", "")))
            f_weights.append(float(item.get("weight", 0.0)))
            f_colors.append("#22c55e")
        for item in neg_feats:
            f_labels.append(_clean_feature_name(item.get("feature", "")))
            f_weights.append(-float(item.get("weight", 0.0)))
            f_colors.append("#ef4444")

        y_f = list(range(len(f_labels)))
        bars_f = ax_feat.barh(y_f, f_weights, color=f_colors)
        ax_feat.axvline(0, color="#334155", linewidth=0.9)
        ax_feat.set_yticks(y_f)
        ax_feat.set_yticklabels(f_labels, fontsize=8)
        ax_feat.invert_yaxis()
        max_w = max((abs(w) for w in f_weights), default=1.0) or 1.0
        ax_feat.set_xlim(-max_w * 1.28, max_w * 1.28)
        for bar, w_val in zip(bars_f, f_weights):
            off = max_w * 0.03
            ha = "left" if w_val >= 0 else "right"
            x_t = w_val + off if w_val >= 0 else w_val - off
            ax_feat.text(x_t, bar.get_y() + bar.get_height() / 2, f"{abs(w_val):.2f}", va="center", ha=ha, fontsize=7.8)
        legend_handles = [
            Patch(facecolor="#22c55e", label="Kéo về Tích cực (POS)"),
            Patch(facecolor="#ef4444", label="Kéo về Tiêu cực (NEG)"),
        ]
        ax_feat.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=8.5)
        ax_feat.set_xlabel("Trọng số Logistic Regression")
        ax_feat.set_title(f"Giải thích mô hình ({dom_lang.upper()}): Đặc trưng ảnh hưởng nhất", pad=28)
        ax_feat.grid(axis="x", linestyle="--", alpha=0.35)
    else:
        ax_feat.text(0.5, 0.5, "Chưa có dữ liệu Top Features", ha="center", va="center", color="#64748b")
        ax_feat.set_axis_off()

    fig.suptitle(_build_suptitle(report, "Phân tích & Giải thích Mô hình — "), fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.subplots_adjust(hspace=0.42, wspace=0.42)
    if save_path is not None:
        out_img = Path(save_path)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_img, dpi=150, bbox_inches="tight")

    return fig
