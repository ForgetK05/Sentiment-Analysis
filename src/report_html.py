from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Sequence

from src.models.classical_ml import EMOTION_VI_MAP, POLARITY_MODEL_META


def _esc(val: Any) -> str:
    return html.escape(str(val if val is not None else ""), quote=True)


def _safe_json_for_script(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return raw.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _clean_feature_label(raw_feat: str) -> str:
    s = str(raw_feat or "")
    if s.startswith("word_tfidf__"):
        return f"Từ: {s[len('word_tfidf__'):]}"
    if s.startswith("char_tfidf__"):
        return f"Ký tự: {s[len('char_tfidf__'):]}"
    if s.startswith("social_lex__"):
        return f"Tín hiệu: {s[len('social_lex__'):]}"
    return s


def _fmt_time(sec: Any) -> str:
    if sec is None or sec == "":
        return "—"
    try:
        val = float(sec)
    except (TypeError, ValueError):
        return "—"
    if val < 1.0:
        return f"{val * 1000.0:.1f} ms"
    return f"{val:.2f} s"


def _compute_agreement_matrix(
    analyzed: Sequence[Any],
    pol_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Tính ma trận % trùng nhãn giữa các mô hình đã chạy trên video."""
    attr_by_key = {
        "tfidf_nb": ("NB", "pred_nb"),
        "tfidf_cnb": ("CNB", "pred_cnb"),
        "tfidf_lr": ("LR", "pred_lr"),
        "tfidf_ridge": ("Ridge", "pred_ridge"),
        "tfidf_sgd": ("SGD", "pred_sgd"),
        "tfidf_svm": ("LinearSVM", "pred_svm"),
        "tfidf_nbsvm": ("NBSVM", "pred_nbsvm"),
        "tfidf_rbf_svm": ("RBF-SVM", "pred_rbf_svm"),
        "tfidf_rf": ("RF", "pred_rf"),
        "tfidf_tree": ("HistGB", "pred_tree"),
        "ml_ensemble": ("ML-Ens", "pred_ml_ensemble"),
        "lexicon": ("Lexicon", "pred_lexicon"),
        "phobert": ("PhoBERT", "pred_phobert"),
        "visobert": ("ViSoBERT", "pred_visobert"),
        "roberta": ("RoBERTa", "pred_roberta"),
        "system_main": ("Chính", "polarity_label"),
    }
    active_cols: list[tuple[str, list[str]]] = []
    seen_keys: set[str] = set()
    for row in pol_rows:
        key = str(row.get("model_key", ""))
        if key in seen_keys or key not in attr_by_key:
            continue
        short_lbl, attr = attr_by_key[key]
        vals = [str(getattr(a, attr, "") or "") for a in analyzed]
        if any(vals):
            seen_keys.add(key)
            active_cols.append((short_lbl, vals))

    if len(active_cols) < 2:
        return {"labels": [], "matrix": []}

    labels = [lbl for lbl, _ in active_cols]
    matrix: list[list[float | None]] = []
    for _, vals_a in active_cols:
        row_out: list[float | None] = []
        for _, vals_b in active_cols:
            pairs = [(a, b) for a, b in zip(vals_a, vals_b) if a and b]
            if not pairs:
                row_out.append(None)
            else:
                same = sum(1 for a, b in pairs if a == b)
                row_out.append(round(same * 100.0 / len(pairs), 1))
        matrix.append(row_out)
    return {"labels": labels, "matrix": matrix}


def _build_family_cards_data(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    summary = report.get("sentiment_summary") or {}
    benchmark = summary.get("classical_ml_benchmark") or {}
    lang_counts = summary.get("language_counts") or {}
    dom_lang = "vi" if int(lang_counts.get("vi", 0) or 0) >= int(lang_counts.get("en", 0) or 0) else "en"
    if dom_lang not in benchmark and benchmark:
        dom_lang = next(iter(benchmark.keys()))
    pol_bench = (benchmark.get(dom_lang) or {}).get("polarity_models") or {}

    comp = report.get("model_comparison") or {}
    pol_rows = comp.get("polarity_models") or []

    extra_meta: dict[str, dict[str, str]] = {
        "lexicon": {
            "family_vi": "Luật & Từ điển (Rule-based)",
            "family_color": "#64748b",
            "desc_vi": "Tra từ điển sắc thái Việt–Anh kết hợp luật đảo chiều phủ định (±2 từ) và từ tăng cường.",
        },
        "phobert": {
            "family_vi": "Deep Learning (Transformer VI)",
            "family_color": "#ec4899",
            "desc_vi": "PhoBERT phân tích ngữ cảnh câu tiếng Việt đã tách từ bằng Underthesea.",
        },
        "visobert": {
            "family_vi": "Deep Learning (Social VI)",
            "family_color": "#db2777",
            "desc_vi": "ViSoBERT huấn luyện chuyên biệt trên bình luận mạng xã hội tiếng Việt.",
        },
        "roberta": {
            "family_vi": "Deep Learning (Transformer EN)",
            "family_color": "#ec4899",
            "desc_vi": "Twitter-RoBERTa chuyên xử lý ngôn ngữ mạng xã hội và tiếng lóng tiếng Anh.",
        },
        "system_main": {
            "family_vi": "Hệ thống chính (Final Decision)",
            "family_color": "#16a34a",
            "desc_vi": "Tổ hợp có trọng số từ các nhánh tốt nhất kèm đồng bộ cảm xúc & ngữ cảnh video.",
        },
    }

    cards: list[dict[str, Any]] = []
    for row in pol_rows:
        m_key = str(row.get("model_key", ""))
        if m_key == "gemma4":
            continue
        meta = POLARITY_MODEL_META.get(m_key) or extra_meta.get(m_key) or {}
        bench_item = pol_bench.get(m_key) or {}
        macro_f1 = bench_item.get("macro_f1")
        cards.append(
            {
                "model_key": m_key,
                "tier": row.get("tier", "base"),
                "model": row.get("model", m_key),
                "family": row.get("family", ""),
                "family_vi": meta.get("family_vi", row.get("family", "")),
                "color": meta.get("family_color", "#2563eb"),
                "desc_vi": meta.get("desc_vi", "Mô hình phân loại sắc thái bình luận."),
                "dom_lang": dom_lang.upper(),
                "bench_macro_f1": round(float(macro_f1) * 100.0, 1) if macro_f1 is not None else None,
                "pos_pct": float(row.get("pos_pct", 0.0)),
                "neu_pct": float(row.get("neu_pct", 0.0)),
                "neg_pct": float(row.get("neg_pct", 0.0)),
                "time_str": _fmt_time(row.get("inference_time_sec")),
                "agreement_gemma": row.get("agreement_vs_gemma_pct"),
            }
        )
    return cards


def _build_xray_items(
    analyzed: Sequence[Any],
    audits: Sequence[Any],
) -> list[dict[str, Any]]:
    audit_map = {int(getattr(g, "comment_index", -1)): g for g in (audits or [])}
    indexed_pairs = sorted(
        enumerate(analyzed),
        key=lambda pair: int(getattr(pair[1], "likes", 0) or 0),
        reverse=True,
    )

    model_fields: tuple[tuple[str, str], ...] = (
        ("CNB (Generative)", "pred_cnb"),
        ("NB (Generative)", "pred_nb"),
        ("LR (Linear)", "pred_lr"),
        ("Ridge (Linear)", "pred_ridge"),
        ("SGD (Linear)", "pred_sgd"),
        ("LinearSVM (SVM)", "pred_svm"),
        ("NBSVM (SVM)", "pred_nbsvm"),
        ("RBF-SVM (SVM)", "pred_rbf_svm"),
        ("RandomForest (Bagging)", "pred_rf"),
        ("HistGB (Boosting)", "pred_tree"),
        ("ML Meta-Ensemble", "pred_ml_ensemble"),
        ("Lexicon (Rule)", "pred_lexicon"),
        ("PhoBERT (DL-VI)", "pred_phobert"),
        ("ViSoBERT (DL-VI)", "pred_visobert"),
        ("RoBERTa (DL-EN)", "pred_roberta"),
    )

    items: list[dict[str, Any]] = []
    for stt, (orig_idx, a) in enumerate(indexed_pairs, start=1):
        votes: list[dict[str, str]] = []
        base_labels: set[str] = set()
        for label_name, attr in model_fields:
            val = str(getattr(a, attr, "") or "").strip()
            if val:
                votes.append({"name": label_name, "pred": val})
                if attr not in {"pred_ml_ensemble"}:
                    base_labels.add(val)

        g = audit_map.get(orig_idx)
        g_audited = bool(g and getattr(g, "gemma_audited", False))
        probs = getattr(a, "polarity_probs", None) or {}
        items.append(
            {
                "stt": stt,
                "username": str(getattr(a, "username", "") or ""),
                "likes": int(getattr(a, "likes", 0) or 0),
                "reply_count": int(getattr(a, "reply_count", 0) or 0),
                "language": str(getattr(a, "language", "vi") or "vi").upper(),
                "original_text": str(getattr(a, "original_text", "") or ""),
                "cleaned_text": str(getattr(a, "cleaned_text", "") or ""),
                "segmented_text": str(getattr(a, "segmented_text", "") or ""),
                "polarity_label": str(getattr(a, "polarity_label", "NEU") or "NEU"),
                "polarity_vi": str(getattr(a, "polarity_vi", "") or ""),
                "polarity_confidence": round(float(getattr(a, "polarity_confidence", 0.0) or 0.0) * 100.0, 1),
                "polarity_score": float(getattr(a, "polarity_score", 0.0) or 0.0),
                "score_POS": round(float(probs.get("POS", 0.0) or 0.0) * 100.0, 1),
                "score_NEU": round(float(probs.get("NEU", 0.0) or 0.0) * 100.0, 1),
                "score_NEG": round(float(probs.get("NEG", 0.0) or 0.0) * 100.0, 1),
                "emotion_label": str(getattr(a, "emotion_label", "Other") or "Other"),
                "emotion_vi": str(getattr(a, "emotion_vi", "") or ""),
                "emotion_confidence": round(float(getattr(a, "emotion_confidence", 0.0) or 0.0) * 100.0, 1),
                "emotion_ml_label": str(getattr(a, "emotion_ml_label", "") or ""),
                "is_on_topic": bool(getattr(a, "is_on_topic", False)),
                "context_relevance": round(float(getattr(a, "context_relevance", 0.0) or 0.0), 3),
                "matched_video_terms": list(getattr(a, "matched_video_terms", []) or []),
                "context_alignment": str(getattr(a, "context_alignment", "") or ""),
                "needs_audit": bool(getattr(a, "needs_audit", False)),
                "has_sarcasm_signal": bool(getattr(a, "has_sarcasm_signal", False)),
                "has_disagreement": len(base_labels) > 1,
                "votes": votes,
                "gemma_audited": g_audited,
                "gemma_polarity": str(getattr(g, "gemma_polarity", "") or "") if g_audited else "",
                "gemma_emotion_vi": str(getattr(g, "gemma_emotion_vi", "") or "") if g_audited else "",
                "gemma_is_sarcasm": bool(getattr(g, "gemma_is_sarcasm", False)) if g_audited else False,
                "gemma_reason": str(getattr(g, "gemma_reason", "") or "") if g_audited else "",
            }
        )
    return items


def generate_html_report(
    report: dict[str, Any],
    preprocessed: Sequence[Any],
    analyzed: Sequence[Any],
    audits: Sequence[Any],
    save_path: str | Path,
) -> Path:
    """Tạo báo cáo HTML tương tác tự chứa (offline) hiển thị luồng pipeline, 5 họ mô hình và Comment X-Ray."""
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    run_name = str(report.get("run_name") or out_file.parent.name or "run")
    media = report.get("media_analysis") or {}
    noise = report.get("noise_filter_summary") or {}
    summary = report.get("sentiment_summary") or {}
    comp = report.get("model_comparison") or {}
    stages = list(report.get("pipeline_stages") or [])
    verdict = report.get("ai_audit_verdict") or {}
    keywords = report.get("keywords_tfidf") or {}

    pol_counts = summary.get("polarity_counts") or {}
    pol_pcts = summary.get("polarity_percentages") or {}
    emo_counts = summary.get("emotion_counts") or {}
    emo_pcts = summary.get("emotion_percentages") or {}
    lang_counts = summary.get("language_counts") or {}
    formula_map = summary.get("ensemble_formula") or {}
    engine_mode = str(summary.get("engine_mode") or "hybrid")
    engine_label = "Classical ML thuần (0% DL)" if engine_mode == "classical" else "Hybrid (Classical ML + Deep Learning)"
    model_scope_label = "Đầy đủ 10+1 mô hình" if summary.get("full_models") else "Tinh gọn 5 đại diện + Ensemble"

    top_emo_key = max(emo_counts, key=emo_counts.get) if emo_counts else "Other"
    top_emo_vi = EMOTION_VI_MAP.get(top_emo_key, top_emo_key)
    top_emo_pct = emo_pcts.get(top_emo_key, 0.0)

    total_time = report.get("total_duration_sec")
    if total_time is None:
        total_time = round(sum(float(s.get("duration_sec") or 0.0) for s in stages), 3)

    # Render Pipeline Stages HTML
    stage_cards_html: list[str] = []
    status_badge_map = {
        "completed": ("ĐÃ CHẠY", "badge-green"),
        "cached": ("DÙNG CACHE", "badge-blue"),
        "skipped": ("BỎ QUA", "badge-gray"),
        "failed": ("LỖI", "badge-red"),
    }
    for idx, st in enumerate(stages):
        st_status = str(st.get("status", "completed")).lower()
        st_text, st_cls = status_badge_map.get(st_status, (st_status.upper(), "badge-gray"))
        metrics_html = "".join(
            f"<li><span>{_esc(k)}:</span> <strong>{_esc(v)}</strong></li>"
            for k, v in (st.get("metrics") or {}).items()
        )
        arrow = '<div class="stage-arrow">→</div>' if idx < len(stages) - 1 else ""
        stage_cards_html.append(
            f"""
            <div class="stage-card">
              <div class="stage-top">
                <span class="stage-id">{_esc(st.get("id", f"GĐ {idx}"))}</span>
                <span class="badge {st_cls}">{_esc(st_text)}</span>
              </div>
              <h4>{_esc(st.get("title", ""))}</h4>
              <div class="stage-time">Thời gian: <strong>{_esc(_fmt_time(st.get("duration_sec")))}</strong></div>
              <p class="stage-summary">{_esc(st.get("summary", ""))}</p>
              {f'<ul class="stage-metrics">{metrics_html}</ul>' if metrics_html else ''}
            </div>
            {arrow}
            """
        )

    # Noise breakdown pills
    reason_vi = {
        "FILTERED_EMPTY": "Rỗng",
        "FILTERED_UNSUPPORTED_LANG": "Ngoại ngữ",
        "FILTERED_URL_OR_SYMBOL": "URL/Số/Emoji",
        "FILTERED_TAG_FRIEND": "Chỉ @tag",
        "FILTERED_NAME_ONLY": "Chỉ tên riêng",
        "FILTERED_SPAM_AD": "Spam/QC",
        "FILTERED_MEANINGLESS": "Vô nghĩa/Seeding",
    }
    noise_pills = "".join(
        f'<span class="pill">{_esc(reason_vi.get(k, k))}: <strong>{int(v)}</strong></span>'
        for k, v in (noise.get("filter_reasons") or {}).items()
    )

    # Formula HTML
    formula_rows = "".join(
        f'<div class="formula-row"><span class="badge badge-blue">{_esc(k.upper())}</span> <code>{_esc(v)}</code></div>'
        for k, v in formula_map.items()
    )

    # Family Cards HTML
    family_cards = _build_family_cards_data(report)
    family_cards_html: list[str] = []
    for c in family_cards:
        bench_pill = (
            f'<span class="pill">Test {c["dom_lang"]} Macro-F1: <strong>{c["bench_macro_f1"]}%</strong></span>'
            if c["bench_macro_f1"] is not None
            else ""
        )
        gemma_pill = (
            f'<span class="pill">Khớp Gemma: <strong>{c["agreement_gemma"]}%</strong></span>'
            if c["agreement_gemma"] is not None
            else ""
        )
        tier_tag = (
            '<span class="badge badge-green">Hệ thống chính</span>'
            if c["tier"] == "system"
            else ('<span class="badge badge-blue">Meta-Ensemble</span>' if c["tier"] == "ensemble" else "")
        )
        family_cards_html.append(
            f"""
            <div class="model-card" style="border-top: 4px solid {_esc(c['color'])}">
              <div class="model-card-head">
                <div>
                  <div class="model-fam">{_esc(c['family_vi'])}</div>
                  <div class="model-name">{_esc(c['model'])}</div>
                </div>
                {tier_tag}
              </div>
              <p class="model-desc">{_esc(c['desc_vi'])}</p>
              <div class="dist-bar" title="POS {c['pos_pct']}% | NEU {c['neu_pct']}% | NEG {c['neg_pct']}%">
                <span class="seg-pos" style="width:{c['pos_pct']}%"></span>
                <span class="seg-neu" style="width:{c['neu_pct']}%"></span>
                <span class="seg-neg" style="width:{c['neg_pct']}%"></span>
              </div>
              <div class="dist-legend">
                <span class="c-pos">POS {c['pos_pct']}%</span>
                <span class="c-neu">NEU {c['neu_pct']}%</span>
                <span class="c-neg">NEG {c['neg_pct']}%</span>
              </div>
              <div class="model-meta">
                <span class="pill">Chạy: <strong>{_esc(c['time_str'])}</strong></span>
                {bench_pill}
                {gemma_pill}
              </div>
            </div>
            """
        )

    # Comparison table rows
    pol_rows = comp.get("polarity_models") or []
    has_gemma_col = any(r.get("agreement_vs_gemma_pct") is not None for r in pol_rows)
    comp_rows_html: list[str] = []
    for r in pol_rows:
        tier = str(r.get("tier", "base"))
        row_cls = "row-system" if tier == "system" else ("row-ensemble" if tier == "ensemble" else "")
        gemma_cells = ""
        if has_gemma_col:
            ag = f"{r['agreement_vs_gemma_pct']}%" if r.get("agreement_vs_gemma_pct") is not None else "—"
            mf1 = f"{r['macro_f1_vs_gemma']:.4f}" if r.get("macro_f1_vs_gemma") is not None else "—"
            kap = f"{r['cohen_kappa_vs_gemma']:.4f}" if r.get("cohen_kappa_vs_gemma") is not None else "—"
            gemma_cells = f"<td>{ag}</td><td>{mf1}</td><td>{kap}</td>"
        comp_rows_html.append(
            f"""
            <tr class="{row_cls}">
              <td><strong>{_esc(r.get('model'))}</strong></td>
              <td>{_esc(r.get('family'))}</td>
              <td>{int(r.get('total_comments', 0))}</td>
              <td class="c-pos">{r.get('pos_pct', 0)}% ({int(r.get('pos_count', 0))})</td>
              <td class="c-neu">{r.get('neu_pct', 0)}% ({int(r.get('neu_count', 0))})</td>
              <td class="c-neg">{r.get('neg_pct', 0)}% ({int(r.get('neg_count', 0))})</td>
              {gemma_cells}
              <td>{_esc(_fmt_time(r.get('inference_time_sec')))}</td>
            </tr>
            """
        )

    # Agreement Matrix HTML
    agr = _compute_agreement_matrix(analyzed, pol_rows)
    agr_html = "<p class='muted'>Chưa đủ số mô hình để tính ma trận đồng thuận.</p>"
    if agr["labels"]:
        th_cells = "".join(f"<th>{_esc(lbl)}</th>" for lbl in agr["labels"])
        body_rows: list[str] = []
        for lbl_r, row_vals in zip(agr["labels"], agr["matrix"]):
            tds: list[str] = [f"<th>{_esc(lbl_r)}</th>"]
            for val in row_vals:
                if val is None:
                    tds.append("<td class='hm-na'>—</td>")
                else:
                    alpha = max(0.08, min(0.75, (float(val) - 35.0) / 85.0))
                    tds.append(
                        f"<td style='background:rgba(37,99,235,{alpha:.2f});"
                        f"color:{'#ffffff' if alpha > 0.45 else '#0f172a'}'>{val:.1f}%</td>"
                    )
            body_rows.append(f"<tr>{''.join(tds)}</tr>")
        agr_html = f"""
        <div class="table-wrap">
          <table class="heatmap-table">
            <thead><tr><th>Mô hình</th>{th_cells}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
          </table>
        </div>
        """

    # Top Features & Keywords HTML
    benchmark = summary.get("classical_ml_benchmark") or {}
    dom_lang = "vi" if int(lang_counts.get("vi", 0) or 0) >= int(lang_counts.get("en", 0) or 0) else "en"
    if dom_lang not in benchmark and benchmark:
        dom_lang = next(iter(benchmark.keys()))
    top_pol_feats = ((benchmark.get(dom_lang) or {}).get("top_features") or {}).get("polarity") or {}
    pos_feats = list(top_pol_feats.get("POS") or [])[:7]
    neg_feats = list(top_pol_feats.get("NEG") or [])[:7]
    max_fw = max(
        [float(x.get("weight", 0.0)) for x in pos_feats + neg_feats] or [1.0]
    ) or 1.0

    def _feat_bars(items: Sequence[dict[str, Any]], bar_cls: str) -> str:
        if not items:
            return "<p class='muted'>Không có dữ liệu.</p>"
        rows: list[str] = []
        for it in items:
            w = float(it.get("weight", 0.0))
            pct = min(100.0, w * 100.0 / max_fw)
            rows.append(
                f"""
                <div class="feat-row">
                  <span class="feat-name">{_esc(_clean_feature_label(str(it.get('feature', ''))))}</span>
                  <div class="feat-track"><div class="feat-fill {bar_cls}" style="width:{pct:.1f}%"></div></div>
                  <span class="feat-val">{w:.2f}</span>
                </div>
                """
            )
        return "".join(rows)

    kw_overall = "".join(
        f'<span class="pill">{_esc(k["keyword"])} <small>({k["tfidf_score"]:.3f})</small></span>'
        for k in (keywords.get("overall") or [])[:10]
    )
    kw_pos = "".join(
        f'<span class="pill pill-pos">{_esc(k["keyword"])}</span>'
        for k in (keywords.get("positive") or [])[:8]
    )
    kw_neg = "".join(
        f'<span class="pill pill-neg">{_esc(k["keyword"])}</span>'
        for k in (keywords.get("negative") or [])[:8]
    )

    # Emotions bar list
    total_valid = max(1, int(summary.get("total_valid_comments", 0) or len(analyzed) or 1))
    emo_bars_html: list[str] = []
    for e_key, e_cnt in sorted(emo_counts.items(), key=lambda kv: kv[1], reverse=True):
        pct = float(emo_pcts.get(e_key, e_cnt * 100.0 / total_valid))
        emo_bars_html.append(
            f"""
            <div class="feat-row">
              <span class="feat-name">{_esc(EMOTION_VI_MAP.get(e_key, e_key))}</span>
              <div class="feat-track"><div class="feat-fill fill-blue" style="width:{min(100.0, pct):.1f}%"></div></div>
              <span class="feat-val">{int(e_cnt)} ({pct:.1f}%)</span>
            </div>
            """
        )

    # Media Context HTML
    entities = ", ".join(str(x) for x in (media.get("doi_tuong_chinh") or [])) or "—"
    media_block_html = (
        f"""
        <div class="media-grid">
          <div><span class="lbl">Chủ đề chính:</span> <strong>{_esc(media.get('chu_de_chinh', '—'))}</strong></div>
          <div><span class="lbl">Mục đích:</span> {_esc(media.get('muc_dich_chinh', '—'))}</div>
          <div><span class="lbl">Sắc thái bài:</span> {_esc(media.get('sac_thai_noi_dung', '—'))}</div>
          <div><span class="lbl">Thực thể chính:</span> {_esc(entities)}</div>
          <div class="full-span"><span class="lbl">Tóm tắt nội dung:</span> {_esc(media.get('tom_tat_ngan', '—'))}</div>
        </div>
        """
        if media
        else "<p class='muted'>Lượt chạy này không có dữ liệu ngữ cảnh video/ảnh (media_analysis=None).</p>"
    )

    verdict_html = ""
    if verdict and verdict.get("audited_count", 0):
        verdict_html = f"""
        <div class="callout">
          <strong>Kết luận Gemma 4 E2B:</strong> {_esc(verdict.get('accuracy_verdict', ''))} |
          Tốt nhất: <strong>{_esc(verdict.get('best_model', '—'))}</strong> |
          Mỉa mai: <strong>{verdict.get('sarcasm_rate_pct', 0)}%</strong> ({verdict.get('sarcasm_detected_count', 0)}/{verdict.get('audited_count', 0)})
          <div style="margin-top:6px">{_esc(verdict.get('summary_vi', ''))}</div>
        </div>
        """

    xray_json = _safe_json_for_script(_build_xray_items(analyzed, audits))

    html_doc = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Báo cáo Phân tích Cảm xúc — {_esc(run_name)}</title>
  <style>
    :root {{
      --bg: #f8fafc; --card: #ffffff; --ink: #0f172a; --muted: #64748b;
      --border: #e2e8f0; --blue: #2563eb; --green: #16a34a; --red: #dc2626;
      --gray: #64748b; --amber: #d97706;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      background: var(--bg); color: var(--ink); line-height: 1.5; }}
    .topbar {{ position: sticky; top: 0; z-index: 20; background: #0f172a; color: #fff;
      padding: 12px 24px; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; }}
    .topbar .title {{ font-weight: 700; font-size: 16px; }}
    .topbar nav a {{ color: #cbd5e1; text-decoration: none; margin-left: 16px; font-size: 13.5px; }}
    .topbar nav a:hover {{ color: #fff; }}
    .container {{ max-width: 1280px; margin: 24px auto; padding: 0 20px 48px; }}
    .section {{ margin-bottom: 28px; }}
    .section-title {{ font-size: 18px; font-weight: 700; margin: 0 0 12px; display: flex; align-items: center; gap: 8px; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .kpi {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }}
    .kpi .k-lbl {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .03em; }}
    .kpi .k-val {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
    .kpi .k-sub {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
    .media-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px 18px; font-size: 14px; }}
    .media-grid .full-span {{ grid-column: 1 / -1; }}
    .lbl {{ color: var(--muted); font-weight: 600; }}
    .muted {{ color: var(--muted); font-size: 13.5px; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
    .badge-green {{ background: #dcfce7; color: #166534; }}
    .badge-blue {{ background: #dbeafe; color: #1e40af; }}
    .badge-red {{ background: #fee2e2; color: #991b1b; }}
    .badge-gray {{ background: #f1f5f9; color: #475569; }}
    .pill {{ display: inline-block; background: #f1f5f9; border: 1px solid var(--border); border-radius: 999px;
      padding: 3px 10px; font-size: 12px; margin: 3px 4px 3px 0; }}
    .pill-pos {{ background: #dcfce7; border-color: #bbf7d0; color: #166534; }}
    .pill-neg {{ background: #fee2e2; border-color: #fecaca; color: #991b1b; }}
    .pipeline-flow {{ display: flex; align-items: stretch; gap: 10px; overflow-x: auto; padding-bottom: 6px; }}
    .stage-card {{ flex: 1 1 210px; min-width: 210px; background: var(--card); border: 1px solid var(--border);
      border-radius: 10px; padding: 14px; display: flex; flex-direction: column; }}
    .stage-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }}
    .stage-id {{ font-size: 12px; font-weight: 700; color: var(--blue); }}
    .stage-card h4 {{ margin: 0 0 4px; font-size: 14.5px; }}
    .stage-time {{ font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
    .stage-summary {{ font-size: 13px; margin: 0 0 8px; color: #334155; }}
    .stage-metrics {{ margin: auto 0 0; padding-left: 16px; font-size: 12px; color: #475569; }}
    .stage-arrow {{ display: flex; align-items: center; font-size: 22px; font-weight: 700; color: #94a3b8; }}
    .formula-box {{ margin-top: 14px; background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 10px; padding: 12px 14px; }}
    .formula-row {{ margin-top: 6px; font-size: 13.5px; }}
    .formula-row code {{ background: #e2e8f0; padding: 2px 6px; border-radius: 6px; font-size: 12.5px; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
    .model-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .model-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }}
    .model-card-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; }}
    .model-fam {{ font-size: 11.5px; font-weight: 700; color: var(--muted); text-transform: uppercase; }}
    .model-name {{ font-size: 14.5px; font-weight: 700; margin-top: 2px; }}
    .model-desc {{ font-size: 12.5px; color: #475569; margin: 8px 0 10px; min-height: 36px; }}
    .dist-bar {{ display: flex; height: 10px; border-radius: 999px; overflow: hidden; background: #e2e8f0; }}
    .seg-pos {{ background: #22c55e; }} .seg-neu {{ background: #94a3b8; }} .seg-neg {{ background: #ef4444; }}
    .dist-legend {{ display: flex; justify-content: space-between; font-size: 11.5px; font-weight: 600; margin: 5px 0 8px; }}
    .c-pos {{ color: var(--green); font-weight: 600; }} .c-neu {{ color: var(--gray); font-weight: 600; }} .c-neg {{ color: var(--red); font-weight: 600; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }}
    th {{ background: #f8fafc; font-weight: 700; color: #334155; }}
    .row-ensemble {{ background: #eff6ff; }}
    .row-system {{ background: #f0fdf4; font-weight: 600; }}
    .heatmap-table th, .heatmap-table td {{ text-align: center; font-size: 12px; padding: 6px 8px; }}
    .heatmap-table tbody th {{ text-align: left; background: #f8fafc; }}
    .hm-na {{ color: #94a3b8; background: #f8fafc; }}
    .feat-row {{ display: grid; grid-template-columns: 150px 1fr 78px; align-items: center; gap: 8px; margin-bottom: 7px; font-size: 12.5px; }}
    .feat-name {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .feat-track {{ background: #f1f5f9; height: 10px; border-radius: 999px; overflow: hidden; }}
    .feat-fill {{ height: 100%; border-radius: 999px; }}
    .fill-pos {{ background: #22c55e; }} .fill-neg {{ background: #ef4444; }} .fill-blue {{ background: #3b82f6; }}
    .feat-val {{ text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }}
    .callout {{ margin-top: 12px; padding: 12px 14px; background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; font-size: 13.5px; }}
    .xray-controls {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 12px; }}
    .xray-controls input {{ flex: 1 1 240px; padding: 8px 12px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 13.5px; }}
    .filter-pills {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .f-btn {{ border: 1px solid #cbd5e1; background: #fff; padding: 5px 11px; border-radius: 999px; cursor: pointer; font-size: 12.5px; }}
    .f-btn.active {{ background: var(--blue); color: #fff; border-color: var(--blue); }}
    .xray-row {{ cursor: pointer; transition: background .12s; }}
    .xray-row:hover {{ background: #f8fafc; }}
    .xray-row td.txt-cell {{ white-space: normal; min-width: 260px; max-width: 480px; }}
    .xray-drawer td {{ white-space: normal; background: #f8fafc; padding: 14px 16px; }}
    .xray-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
    .vote-grid {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }}
    .vote-chip {{ border: 1px solid var(--border); background: #fff; border-radius: 8px; padding: 4px 8px; font-size: 12px; }}
    .lbl-POS {{ color: #166534; background: #dcfce7; padding: 1px 6px; border-radius: 5px; font-weight: 700; }}
    .lbl-NEU {{ color: #475569; background: #e2e8f0; padding: 1px 6px; border-radius: 5px; font-weight: 700; }}
    .lbl-NEG {{ color: #991b1b; background: #fee2e2; padding: 1px 6px; border-radius: 5px; font-weight: 700; }}
    .pager {{ display: flex; justify-content: space-between; align-items: center; margin-top: 12px; font-size: 13px; }}
    .pager button {{ padding: 5px 12px; border: 1px solid #cbd5e1; background: #fff; border-radius: 6px; cursor: pointer; }}
    .pager button:disabled {{ opacity: 0.45; cursor: not-allowed; }}
  </style>
</head>
<body>
  <header class="topbar">
    <div class="title">Báo cáo Phân tích Cảm xúc TikTok — <code>{_esc(run_name)}</code></div>
    <nav>
      <a href="#overview">Tổng quan</a>
      <a href="#pipeline">Luồng hệ thống</a>
      <a href="#models">5 Họ Mô hình</a>
      <a href="#xray">Comment X-Ray</a>
    </nav>
  </header>

  <main class="container">
    <section id="overview" class="section">
      <h2 class="section-title">1. Tổng quan &amp; Ngữ cảnh Video</h2>
      <div class="kpi-grid">
        <div class="kpi">
          <div class="k-lbl">Bình luận hợp lệ</div>
          <div class="k-val">{int(noise.get('valid_comments', len(analyzed)))} / {int(noise.get('total_raw_comments', len(preprocessed)))}</div>
          <div class="k-sub">VI: {int(lang_counts.get('vi', 0))} | EN: {int(lang_counts.get('en', 0))} | Lọc: {int(noise.get('filtered_noise_count', 0))}</div>
        </div>
        <div class="kpi">
          <div class="k-lbl">Xu hướng sắc thái</div>
          <div class="k-val">{_esc(summary.get('overall_tendency', '—'))}</div>
          <div class="k-sub">POS {pol_pcts.get('POS', 0)}% | NEU {pol_pcts.get('NEU', 0)}% | NEG {pol_pcts.get('NEG', 0)}%</div>
        </div>
        <div class="kpi">
          <div class="k-lbl">Cảm xúc chủ đạo</div>
          <div class="k-val">{_esc(top_emo_vi)}</div>
          <div class="k-sub">Chiếm {top_emo_pct}% | Tự tin TB: {round(float(summary.get('avg_confidence', 0.0) or 0.0) * 100.0, 1)}%</div>
        </div>
        <div class="kpi">
          <div class="k-lbl">Cấu hình chạy</div>
          <div class="k-val" style="font-size:16px">{_esc(engine_label)}</div>
          <div class="k-sub">{_esc(model_scope_label)} • {_esc(_fmt_time(total_time))}</div>
        </div>
      </div>

      <div class="card">
        {media_block_html}
        {verdict_html}
      </div>
    </section>

    <section id="pipeline" class="section">
      <h2 class="section-title">2. Cách Hệ thống Chạy (Live Pipeline Funnel)</h2>
      <div class="pipeline-flow">
        {''.join(stage_cards_html)}
      </div>
      <div class="card" style="margin-top:12px">
        <div style="font-size:13.5px">
          <strong>Chi tiết lọc nhiễu:</strong>
          {noise_pills if noise_pills else '<span class="muted">Không có bình luận nhiễu.</span>'}
        </div>
        {f'<div class="formula-box"><strong>Công thức tổ hợp điểm cuối (Stage 3):</strong>{formula_rows}</div>' if formula_rows else ''}
      </div>
    </section>

    <section id="models" class="section">
      <h2 class="section-title">3. Biểu diễn 5 Họ Mô hình &amp; Giải thích</h2>
      <div class="model-cards">
        {''.join(family_cards_html)}
      </div>

      <div class="card section">
        <h3 style="margin-top:0;font-size:15.5px">Bảng so sánh chi tiết các mô hình trên video hiện tại</h3>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Mô hình</th><th>Nhóm phương pháp</th><th>Số mẫu</th>
                <th>Tích cực (POS)</th><th>Trung tính (NEU)</th><th>Tiêu cực (NEG)</th>
                {'<th>Khớp Gemma</th><th>Macro-F1</th><th>Kappa</th>' if has_gemma_col else ''}
                <th>Thời gian</th>
              </tr>
            </thead>
            <tbody>{''.join(comp_rows_html)}</tbody>
          </table>
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <h3 style="margin-top:0;font-size:15.5px">Ma trận đồng thuận giữa các mô hình (%)</h3>
          <p class="muted" style="margin-top:-4px">Tỷ lệ % số bình luận mà hai mô hình đưa ra cùng một nhãn POS / NEU / NEG.</p>
          {agr_html}
        </div>
        <div class="card">
          <h3 style="margin-top:0;font-size:15.5px">Phân bố 7 cung bậc cảm xúc</h3>
          {''.join(emo_bars_html)}
        </div>
      </div>

      <div class="grid-2" style="margin-top:16px">
        <div class="card">
          <h3 style="margin-top:0;font-size:15.5px">Giải thích mô hình ({_esc(dom_lang.upper())}): Top đặc trưng kéo về POS / NEG</h3>
          <div style="margin-bottom:10px"><strong class="c-pos">Kéo về Tích cực (POS)</strong></div>
          {_feat_bars(pos_feats, 'fill-pos')}
          <div style="margin:12px 0 10px"><strong class="c-neg">Kéo về Tiêu cực (NEG)</strong></div>
          {_feat_bars(neg_feats, 'fill-neg')}
        </div>
        <div class="card">
          <h3 style="margin-top:0;font-size:15.5px">Từ khóa nổi bật trong bình luận (TF-IDF)</h3>
          <div style="margin-bottom:8px"><strong>Chung toàn video:</strong><br/>{kw_overall or '<span class="muted">—</span>'}</div>
          <div style="margin-bottom:8px"><strong class="c-pos">Nhóm Tích cực:</strong><br/>{kw_pos or '<span class="muted">—</span>'}</div>
          <div><strong class="c-neg">Nhóm Tiêu cực:</strong><br/>{kw_neg or '<span class="muted">—</span>'}</div>
        </div>
      </div>
    </section>

    <section id="xray" class="section">
      <h2 class="section-title">4. Comment X-Ray Explorer (Bấm từng dòng để soi phiếu bầu mô hình)</h2>
      <div class="card">
        <div class="xray-controls">
          <input id="xraySearch" type="text" placeholder="Tìm nội dung bình luận, @username hoặc từ khóa..." />
          <div id="xrayFilters" class="filter-pills">
            <button class="f-btn active" data-filter="all">Tất cả</button>
            <button class="f-btn" data-filter="POS">Tích cực</button>
            <button class="f-btn" data-filter="NEU">Trung tính</button>
            <button class="f-btn" data-filter="NEG">Tiêu cực</button>
            <button class="f-btn" data-filter="needs_audit">Ca khó (needs_audit)</button>
            <button class="f-btn" data-filter="sarcasm">Nghi mỉa mai</button>
            <button class="f-btn" data-filter="disagree">Mô hình bất đồng</button>
            <button class="f-btn" data-filter="on_topic">Bám sát video</button>
          </div>
        </div>
        <div class="muted" id="xrayCount" style="margin-bottom:8px"></div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th><th>Bình luận</th><th>Sắc thái</th><th>Cảm xúc</th>
                <th>Likes</th><th>Ngôn ngữ</th><th>Cờ</th>
              </tr>
            </thead>
            <tbody id="xrayBody"></tbody>
          </table>
        </div>
        <div class="pager">
          <button id="prevPage">← Trước</button>
          <span id="pageInfo"></span>
          <button id="nextPage">Sau →</button>
        </div>
      </div>
    </section>
  </main>

  <script id="xray-data" type="application/json">{xray_json}</script>
  <script>
    (function () {{
      const rawData = JSON.parse(document.getElementById('xray-data').textContent || '[]');
      const PAGE_SIZE = 25;
      let currentFilter = 'all';
      let currentQuery = '';
      let currentPage = 1;
      let openStt = null;

      function esc(str) {{
        return String(str == null ? '' : str)
          .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
      }}

      function getFiltered() {{
        const q = currentQuery.trim().toLowerCase();
        return rawData.filter(item => {{
          if (currentFilter === 'POS' && item.polarity_label !== 'POS') return false;
          if (currentFilter === 'NEU' && item.polarity_label !== 'NEU') return false;
          if (currentFilter === 'NEG' && item.polarity_label !== 'NEG') return false;
          if (currentFilter === 'needs_audit' && !item.needs_audit) return false;
          if (currentFilter === 'sarcasm' && !item.has_sarcasm_signal && !item.gemma_is_sarcasm) return false;
          if (currentFilter === 'disagree' && !item.has_disagreement) return false;
          if (currentFilter === 'on_topic' && !item.is_on_topic) return false;
          if (!q) return true;
          return (
            item.original_text.toLowerCase().includes(q) ||
            item.username.toLowerCase().includes(q) ||
            item.cleaned_text.toLowerCase().includes(q)
          );
        }});
      }}

      function renderDrawer(item) {{
        const votesHtml = item.votes.map(v =>
          `<div class="vote-chip"><span>${{esc(v.name)}}:</span> <span class="lbl-${{esc(v.pred)}}">${{esc(v.pred)}}</span></div>`
        ).join('');
        const terms = item.matched_video_terms.length ? item.matched_video_terms.join(', ') : '—';
        const gemmaBlock = item.gemma_audited
          ? `<div style="margin-top:8px"><strong>Gemma 4 E2B Audit:</strong> <span class="lbl-${{esc(item.gemma_polarity)}}">${{esc(item.gemma_polarity)}}</span> (${{esc(item.gemma_emotion_vi)}}) — ${{esc(item.gemma_reason)}}</div>`
          : '';
        return `
          <tr class="xray-drawer"><td colspan="7">
            <div class="xray-grid">
              <div>
                <div><strong>1. Tiền xử lý &amp; Ngữ cảnh video</strong></div>
                <div style="margin-top:4px;font-size:12.5px">
                  <div><span class="lbl">Làm sạch:</span> ${{esc(item.cleaned_text)}}</div>
                  <div><span class="lbl">Tách từ:</span> ${{esc(item.segmented_text)}}</div>
                  <div><span class="lbl">Liên quan video:</span> ${{item.is_on_topic ? 'Có' : 'Không'}} (score ${{item.context_relevance}}) | <span class="lbl">Từ khóa khớp:</span> ${{esc(terms)}}</div>
                  <div><span class="lbl">Context alignment:</span> ${{esc(item.context_alignment || '—')}}</div>
                </div>
              </div>
              <div>
                <div><strong>2. Kết quả tổ hợp cuối</strong></div>
                <div style="margin-top:6px">
                  <div class="dist-bar" style="height:12px">
                    <span class="seg-pos" style="width:${{item.score_POS}}%"></span>
                    <span class="seg-neu" style="width:${{item.score_NEU}}%"></span>
                    <span class="seg-neg" style="width:${{item.score_NEG}}%"></span>
                  </div>
                  <div class="dist-legend">
                    <span class="c-pos">POS ${{item.score_POS}}%</span>
                    <span class="c-neu">NEU ${{item.score_NEU}}%</span>
                    <span class="c-neg">NEG ${{item.score_NEG}}%</span>
                  </div>
                  <div style="font-size:12.5px">
                    Điểm cực tính: <strong>${{item.polarity_score}}</strong> |
                    Cảm xúc cuối: <strong>${{esc(item.emotion_vi)}}</strong> (${{item.emotion_confidence}}%)
                  </div>
                  ${{gemmaBlock}}
                </div>
              </div>
            </div>
            <div style="margin-top:10px">
              <strong>3. Phiếu bầu từng mô hình theo họ:</strong>
              <div class="vote-grid">${{votesHtml}}</div>
            </div>
          </td></tr>
        `;
      }}

      function render() {{
        const filtered = getFiltered();
        const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
        if (currentPage > totalPages) currentPage = totalPages;
        const start = (currentPage - 1) * PAGE_SIZE;
        const pageItems = filtered.slice(start, start + PAGE_SIZE);

        document.getElementById('xrayCount').textContent =
          `Hiển thị ${{pageItems.length}} / ${{filtered.length}} bình luận khớp điều kiện (Tổng ${{rawData.length}}). Bấm vào dòng bất kỳ để mở X-Ray.`;
        document.getElementById('pageInfo').textContent = `Trang ${{currentPage}} / ${{totalPages}}`;
        document.getElementById('prevPage').disabled = currentPage <= 1;
        document.getElementById('nextPage').disabled = currentPage >= totalPages;

        const tbody = document.getElementById('xrayBody');
        tbody.innerHTML = pageItems.map(item => {{
          const flags = [];
          if (item.needs_audit) flags.push('<span class="badge badge-blue">Ca khó</span>');
          if (item.has_sarcasm_signal || item.gemma_is_sarcasm) flags.push('<span class="badge badge-red">Mỉa mai</span>');
          if (item.has_disagreement) flags.push('<span class="badge badge-gray">Bất đồng</span>');
          if (item.is_on_topic) flags.push('<span class="badge badge-green">Bám video</span>');
          const rowHtml = `
            <tr class="xray-row" data-stt="${{item.stt}}">
              <td>${{item.stt}}</td>
              <td class="txt-cell"><strong>@${{esc(item.username || 'user')}}</strong>: ${{esc(item.original_text)}}</td>
              <td><span class="lbl-${{esc(item.polarity_label)}}">${{esc(item.polarity_vi)}}</span> <small>(${{item.polarity_confidence}}%)</small></td>
              <td>${{esc(item.emotion_vi)}} <small>(${{item.emotion_confidence}}%)</small></td>
              <td>${{item.likes}}</td>
              <td>${{esc(item.language)}}</td>
              <td>${{flags.join(' ') || '—'}}</td>
            </tr>
          `;
          return openStt === item.stt ? rowHtml + renderDrawer(item) : rowHtml;
        }}).join('');
      }}

      document.getElementById('xrayBody').addEventListener('click', function (e) {{
        const tr = e.target.closest('tr.xray-row');
        if (!tr) return;
        const stt = Number(tr.dataset.stt);
        openStt = openStt === stt ? null : stt;
        render();
      }});

      document.getElementById('xraySearch').addEventListener('input', function (e) {{
        currentQuery = e.target.value;
        currentPage = 1;
        render();
      }});

      document.getElementById('xrayFilters').addEventListener('click', function (e) {{
        const btn = e.target.closest('button.f-btn');
        if (!btn) return;
        document.querySelectorAll('#xrayFilters .f-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFilter = btn.dataset.filter;
        currentPage = 1;
        render();
      }});

      document.getElementById('prevPage').addEventListener('click', function () {{
        if (currentPage > 1) {{ currentPage--; render(); }}
      }});
      document.getElementById('nextPage').addEventListener('click', function () {{
        currentPage++; render();
      }});

      render();
    }})();
  </script>
</body>
</html>
"""
    out_file.write_text(html_doc, encoding="utf-8")
    return out_file
