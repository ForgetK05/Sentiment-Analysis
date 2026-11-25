from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.models.classical_ml import EMOTION_LABELS, EMOTION_VI_MAP

# Đảm bảo font tiếng Việt hiển thị sạch trên biểu đồ Windows
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False


def plot_sentiment_dashboard(
    report: dict[str, Any],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Vẽ dashboard 2x2: Polarity, 7 Cảm xúc, Bám sát nội dung video (hoặc Ngôn ngữ) & So sánh mô hình."""
    summary = report.get("sentiment_summary")
    if not summary or summary.get("total_valid_comments", 0) == 0:
        raise ValueError("Báo cáo chưa có dữ liệu thống kê bình luận để vẽ biểu đồ.")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_pol, ax_emo = axes[0, 0], axes[0, 1]
    ax_ctx, ax_comp = axes[1, 0], axes[1, 1]

    # 1. Polarity: Hệ thống Hybrid vs Gemma 4 (nếu có)
    pol_counts = summary["polarity_counts"]
    pol_labels = ["Tích cực (POS)", "Trung tính (NEU)", "Tiêu cực (NEG)"]
    sys_vals = [pol_counts.get("POS", 0), pol_counts.get("NEU", 0), pol_counts.get("NEG", 0)]

    comp = report.get("model_comparison") or {}
    pol_models = comp.get("polarity_models") or []
    gemma_row = next(
        (r for r in pol_models if "Gemma 4" in str(r.get("model", ""))),
        None,
    )

    x_idx = [0, 1, 2]
    if gemma_row:
        g_vals = [gemma_row["pos_count"], gemma_row["neu_count"], gemma_row["neg_count"]]
        w = 0.36
        b1 = ax_pol.bar(
            [x - w / 2 for x in x_idx], sys_vals, width=w, label="Hybrid System", color="#3498db"
        )
        b2 = ax_pol.bar(
            [x + w / 2 for x in x_idx], g_vals, width=w, label="Gemma 4 E2B", color="#e67e22"
        )
        ax_pol.legend()
        for bar in list(b1) + list(b2):
            ax_pol.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(int(bar.get_height())),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax_pol.set_title("Sắc thái Polarity: Hybrid System vs Gemma 4 E2B")
    else:
        colors = ["#2ecc71", "#95a5a6", "#e74c3c"]
        bars = ax_pol.bar(x_idx, sys_vals, color=colors)
        for bar in bars:
            ax_pol.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(int(bar.get_height())),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax_pol.set_title(f"Phân bố Sắc thái ({summary.get('overall_tendency', '')})")

    ax_pol.set_xticks(x_idx)
    ax_pol.set_xticklabels(pol_labels)
    ax_pol.set_ylabel("Số bình luận")
    ax_pol.grid(axis="y", linestyle="--", alpha=0.35)

    # 2. 7 Cảm xúc chi tiết
    emo_counts = summary["emotion_counts"]
    emo_items = [(EMOTION_VI_MAP[e], emo_counts.get(e, 0)) for e in EMOTION_LABELS]
    emo_items.sort(key=lambda x: x[1], reverse=True)
    emo_names = [k for k, _ in emo_items]
    emo_vals = [v for _, v in emo_items]
    emo_colors = ["#e74c3c", "#9b59b6", "#34495e", "#2ecc71", "#f39c12", "#1abc9c", "#95a5a6"]
    bars_e = ax_emo.barh(emo_names, emo_vals, color=emo_colors[: len(emo_names)])
    ax_emo.invert_yaxis()
    ax_emo.set_xlabel("Số bình luận")
    ax_emo.set_title("Phân bố 7 cung bậc cảm xúc")
    ax_emo.grid(axis="x", linestyle="--", alpha=0.35)
    max_e = max(emo_vals, default=1)
    for bar, val in zip(bars_e, emo_vals):
        ax_emo.text(
            bar.get_width() + max_e * 0.02,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
        )

    # 3. Phân bố sắc thái theo mức độ bám sát nội dung video (hoặc fallback Ngôn ngữ)
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
        b_on = ax_ctx.bar(
            [x - w_ctx / 2 for x in ctx_x],
            on_vals,
            width=w_ctx,
            label=f"Bám sát video ({on_topic_info.get('on_topic_count', 0)})",
            color="#2980b9",
        )
        b_off = ax_ctx.bar(
            [x + w_ctx / 2 for x in ctx_x],
            off_vals,
            width=w_ctx,
            label=f"Ngoài lề ({on_topic_info.get('off_topic_count', 0)})",
            color="#bdc3c7",
        )
        for bar in list(b_on) + list(b_off):
            ax_ctx.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(int(bar.get_height())),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax_ctx.set_xticks(ctx_x)
        ax_ctx.set_xticklabels(["Tích cực (POS)", "Trung tính (NEU)", "Tiêu cực (NEG)"])
        ax_ctx.set_ylabel("Số bình luận")
        ax_ctx.set_title(
            f"Bám sát nội dung video: {on_topic_info.get('on_topic_pct', 0)}% "
            f"({on_topic_info.get('on_topic_count', 0)}/{summary.get('total_valid_comments', 0)})"
        )
        ax_ctx.legend(fontsize=8.5)
        ax_ctx.grid(axis="y", linestyle="--", alpha=0.35)
    else:
        lang_map = summary.get("language_counts") or {}
        lang_labels = ["Tiếng Việt (vi)", "Tiếng Anh (en)"]
        lang_vals = [int(lang_map.get("vi", 0)), int(lang_map.get("en", 0))]
        bars_l = ax_ctx.bar(lang_labels, lang_vals, color=["#1abc9c", "#3498db"])
        for bar in bars_l:
            ax_ctx.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(int(bar.get_height())),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax_ctx.set_ylabel("Số bình luận")
        ax_ctx.set_title(
            f"Phân bố ngôn ngữ (Cực tính TB: {summary.get('avg_polarity_score', 0)} | "
            f"Tự tin TB: {summary.get('avg_confidence', 0)})"
        )
        ax_ctx.grid(axis="y", linestyle="--", alpha=0.35)

    # 4. So sánh các mô hình ML cổ điển vs Deep Learning vs Hybrid
    non_ref_models = [r for r in pol_models if "Gemma 4" not in str(r.get("model", ""))]
    if non_ref_models and non_ref_models[0].get("agreement_vs_gemma_pct") is not None:
        m_names = [r["model"].replace(" (Hệ thống chính)", "\n(Hybrid)") for r in non_ref_models]
        agrees = [float(r.get("agreement_vs_gemma_pct") or 0.0) for r in non_ref_models]
        f1s = [float(r.get("macro_f1_vs_gemma") or 0.0) * 100.0 for r in non_ref_models]
        y_pos = list(range(len(m_names)))
        h = 0.36
        ax_comp.barh(
            [y - h / 2 for y in y_pos], agrees, height=h, label="Agreement (%) vs Gemma 4", color="#2980b9"
        )
        ax_comp.barh(
            [y + h / 2 for y in y_pos], f1s, height=h, label="Macro-F1 (%) vs Gemma 4", color="#8e44ad"
        )
        ax_comp.set_yticks(y_pos)
        ax_comp.set_yticklabels(m_names, fontsize=8.5)
        ax_comp.invert_yaxis()
        ax_comp.set_xlim(0, 108)
        ax_comp.set_xlabel("Tỷ lệ khớp thực tế (%)")
        ax_comp.set_title("So sánh mô hình ML cổ điển vs Deep Learning vs Hybrid")
        ax_comp.legend(fontsize=8)
        ax_comp.grid(axis="x", linestyle="--", alpha=0.35)
    elif non_ref_models:
        m_names = [r["model"].replace(" (Hệ thống chính)", "\n(Hybrid)") for r in non_ref_models]
        pos_p = [float(r["pos_pct"]) for r in non_ref_models]
        neu_p = [float(r["neu_pct"]) for r in non_ref_models]
        neg_p = [float(r["neg_pct"]) for r in non_ref_models]
        y_pos = list(range(len(m_names)))
        ax_comp.barh(y_pos, pos_p, label="POS %", color="#2ecc71")
        ax_comp.barh(y_pos, neu_p, left=pos_p, label="NEU %", color="#95a5a6")
        left_neg = [a + b for a, b in zip(pos_p, neu_p)]
        ax_comp.barh(y_pos, neg_p, left=left_neg, label="NEG %", color="#e74c3c")
        ax_comp.set_yticks(y_pos)
        ax_comp.set_yticklabels(m_names, fontsize=8.5)
        ax_comp.invert_yaxis()
        ax_comp.set_xlim(0, 100)
        ax_comp.set_xlabel("Phân bố (%)")
        ax_comp.set_title("Phân bố dự đoán giữa các mô hình ML & DL")
        ax_comp.legend(fontsize=8)

    media_info = report.get("media_analysis") or {}
    topic = media_info.get("chu_de_chinh")
    noise_info = report.get("noise_filter_summary") or {}
    lang_counts = summary.get("language_counts") or {}
    lang_str = f"VI: {lang_counts.get('vi', 0)} | EN: {lang_counts.get('en', 0)}"
    subtitle = (
        f"Hợp lệ: {noise_info.get('valid_comments', summary.get('total_valid_comments', 0))} "
        f"({lang_str}) | Đã lọc nhiễu: {noise_info.get('filtered_noise_count', 0)}"
    )
    if topic:
        fig.suptitle(f"Chủ đề: {topic}\n({subtitle})", fontsize=13, fontweight="bold")
    else:
        fig.suptitle(f"Dashboard Phân tích Cảm xúc\n({subtitle})", fontsize=13, fontweight="bold")

    fig.tight_layout()
    if save_path is not None:
        out_img = Path(save_path)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_img, dpi=150, bbox_inches="tight")

    return fig
