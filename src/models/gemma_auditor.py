from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import requests
from sklearn.metrics import cohen_kappa_score, f1_score
from tqdm import tqdm

from src.models.classical_ml import EMOTION_LABELS, EMOTION_VI_MAP
from src.models.media_context import MediaAnalysisResult
from src.models.sentiment_engine import AnalyzedComment

BATCH_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "gemma_polarity": {
                        "type": "string",
                        "enum": ["POS", "NEG", "NEU"],
                    },
                    "gemma_emotion": {
                        "type": "string",
                        "enum": list(EMOTION_LABELS),
                    },
                    "gemma_is_sarcasm": {"type": "boolean"},
                    "gemma_reason": {"type": "string"},
                },
                "required": [
                    "id",
                    "gemma_polarity",
                    "gemma_emotion",
                    "gemma_is_sarcasm",
                    "gemma_reason",
                ],
            },
        }
    },
    "required": ["items"],
}

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "accuracy_verdict": {
            "type": "string",
            "enum": [
                "Chính xác cao",
                "Khá chính xác",
                "Trung bình (cần cải thiện)",
                "Chưa chính xác",
            ],
        },
        "best_model": {"type": "string"},
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
        },
        "weaknesses": {
            "type": "array",
            "items": {"type": "string"},
        },
        "summary_vi": {"type": "string"},
    },
    "required": [
        "accuracy_verdict",
        "best_model",
        "strengths",
        "weaknesses",
        "summary_vi",
    ],
}

_AUDIT_SYSTEM_PROMPT = (
    "Bạn là giám khảo độc lập chuyên thẩm định cảm xúc bình luận TikTok (hỗ trợ cả Tiếng Việt và Tiếng Anh).\n"
    "Hãy đọc kỹ BỐI CẢNH VIDEO và đánh giá cảm xúc thực sự của từng bình luận một cách cân bằng, khách quan:\n"
    "1. PHÂN BIỆT RÕ 3 NHÃN POLARITY:\n"
    "   - POS: Khen ngợi, hào hứng, yêu thích, ủng hộ, đồng cảm tích cực "
    "(lưu ý: từ lóng tiếng Anh như 'peak', 'fire', 'smooth', 'holy shit this is peak', 'so freaking good' là POS).\n"
    "   - NEU: Câu hỏi, bình luận mô tả trung lập, suy ngẫm chung chung, đùa vui nhẹ không khen không chê, "
    "hoặc câu quá mơ hồ/ngắn không rõ thái độ. TUYỆT ĐỐI KHÔNG gán các câu trung lập/hỏi/mô tả này thành NEG.\n"
    "   - NEG: Chê bai, ghét, tức giận, buồn bã, sợ hãi, thất vọng, khinh thường, hoặc mỉa mai tiêu cực.\n"
    "2. PHÁT HIỆN NÓI MỈA / CHÂM BIẾM (`gemma_is_sarcasm`):\n"
    "   - Ví dụ: dùng emoji cười 🤣😂:)) nhưng nội dung thực chất đang cà khịa/chê bai.\n"
    "3. CẢM XÚC CHI TIẾT (`gemma_emotion`):\n"
    "   - Anger (tức giận), Sadness (buồn bã), Fear (lo lắng/sợ), Enjoyment (vui vẻ/hài lòng), "
    "Disgust (chán ghét/khó chịu), Surprise (ngạc nhiên), Other (trung tính/khác).\n"
    "4. Giải thích `gemma_reason` ngắn gọn 1 câu bằng tiếng Việt và trả về ĐẦY ĐỦ tất cả `id`."
)


@dataclass
class GemmaAuditItem:
    """Kết quả thẩm định thực tế của Gemma 4 E2B cho 1 bình luận."""

    comment_index: int
    gemma_audited: bool
    gemma_polarity: str
    gemma_emotion: str
    gemma_emotion_vi: str
    gemma_is_sarcasm: bool
    gemma_agrees_polarity: bool | None
    gemma_agrees_emotion: bool | None
    gemma_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _format_video_context(media_result: MediaAnalysisResult | dict[str, Any] | None) -> str:
    if media_result is None:
        return "Không có thông tin video (đánh giá thuần theo nội dung bình luận)."
    data = media_result.to_dict() if hasattr(media_result, "to_dict") else dict(media_result)
    doi_tuong = ", ".join(data.get("doi_tuong_chinh") or []) or "Chưa xác định"
    lines = [
        f"- Chủ đề chính: {data.get('chu_de_chinh', '')}",
        f"- Mục đích bài đăng: {data.get('muc_dich_chinh', '')}",
        f"- Quan điểm nội dung: {data.get('quan_diem_noi_dung', '')}",
        f"- Đối tượng chính: {doi_tuong}",
        f"- Sắc thái bài đăng: {data.get('sac_thai_noi_dung', '')}",
        f"- Tóm tắt: {data.get('tom_tat_ngan', '')}",
    ]
    key_points = [str(p).strip() for p in (data.get("chi_tiet_va_luan_diem") or []) if str(p).strip()][:5]
    if key_points:
        lines.append("- Chi tiết & luận điểm chính: " + " | ".join(key_points))
    note = str(data.get("luu_y_danh_gia_comment") or "").strip()
    if note:
        lines.append(f"- Lưu ý diễn giải bình luận: {note}")
    return "\n".join(lines)


def _extract_json_dict(raw_text: str) -> dict[str, Any] | None:
    cleaned = (raw_text or "").strip()
    if not cleaned:
        return None
    fenced = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL
    ).strip()
    for candidate in (cleaned, fenced):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", fenced, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _normalize_gemma_polarity(pol: str, is_sarcasm: bool) -> str:
    """Chuẩn hóa nhãn cực tính từ Gemma về POS/NEG/NEU."""
    pol = pol.strip().upper()
    if pol not in {"POS", "NEG", "NEU"}:
        pol = "NEU"
    if is_sarcasm and pol == "POS":
        return "NEG"
    return pol


class GemmaAuditor:
    """Giai đoạn 4 (AI Đầu ra): Gemma 4 E2B thẩm định thực tế & so sánh các mô hình ML/DL."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "gemma4-e2b-opt",
        batch_size: int = 8,
        temperature: float = 0.1,
        timeout: int = 180,
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.temperature = float(temperature)
        self.timeout = timeout

    def _call_ollama_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
                "num_ctx": 16384,
                "repeat_penalty": 1.05,
            },
            "messages": messages,
        }
        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        raw_content = resp.json().get("message", {}).get("content", "")
        parsed = _extract_json_dict(raw_content)
        if parsed is None:
            raise ValueError(f"Gemma 4 trả về JSON không hợp lệ: {raw_content[:200]}")
        return parsed

    def _audit_chunk(
        self,
        chunk_pairs: list[tuple[int, AnalyzedComment]],
        context_block: str,
    ) -> dict[int, dict[str, Any]]:
        comment_lines = [
            f"[{idx}] ({item.language.upper()}) {item.original_text}"
            for idx, item in chunk_pairs
        ]
        user_prompt = (
            f"BỐI CẢNH VIDEO/BÀI ĐĂNG:\n{context_block}\n\n"
            "DANH SÁCH BÌNH LUẬN CẦN ĐÁNH GIÁ ĐỘC LẬP:\n"
            + "\n".join(comment_lines)
            + "\n\nHãy đánh giá từng bình luận theo đúng `id` trong ngoặc vuông `[id]`."
        )
        messages = [
            {"role": "system", "content": _AUDIT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        parsed_map: dict[int, dict[str, Any]] = {}
        try:
            resp_obj = self._call_ollama_json(messages, BATCH_AUDIT_SCHEMA)
            for row in resp_obj.get("items") or []:
                if not isinstance(row, dict) or "id" not in row:
                    continue
                cid = int(row["id"])
                raw_pol = str(row.get("gemma_polarity", "NEU"))
                emo = str(row.get("gemma_emotion", "Other")).strip()
                if emo not in EMOTION_LABELS:
                    emo = "Other"
                is_sarcasm = bool(row.get("gemma_is_sarcasm", False))
                pol = _normalize_gemma_polarity(raw_pol, is_sarcasm)

                parsed_map[cid] = {
                    "gemma_polarity": pol,
                    "gemma_emotion": emo,
                    "gemma_is_sarcasm": is_sarcasm,
                    "gemma_reason": str(row.get("gemma_reason", "")).strip(),
                }
        except Exception as exc:
            if len(chunk_pairs) > 1:
                mid = len(chunk_pairs) // 2
                parsed_map.update(self._audit_chunk(chunk_pairs[:mid], context_block))
                parsed_map.update(self._audit_chunk(chunk_pairs[mid:], context_block))
                return parsed_map
            print(f"[Cảnh báo] Lỗi thẩm định Gemma 4 cho comment #{chunk_pairs[0][0]}: {exc}")
            return {}

        missing = [pair for pair in chunk_pairs if pair[0] not in parsed_map]
        if missing and len(chunk_pairs) > 1:
            if len(missing) == len(chunk_pairs):
                mid = len(missing) // 2
                parsed_map.update(self._audit_chunk(missing[:mid], context_block))
                parsed_map.update(self._audit_chunk(missing[mid:], context_block))
            else:
                parsed_map.update(self._audit_chunk(missing, context_block))
        return parsed_map

    def audit_comments(
        self,
        analyzed: Sequence[AnalyzedComment],
        media_result: MediaAnalysisResult | dict[str, Any] | None = None,
        audit_limit: int | None = None,
        show_progress: bool = True,
    ) -> list[GemmaAuditItem]:
        """Dùng Gemma 4 E2B đánh giá thực tế toàn bộ (hoặc `audit_limit`) bình luận hợp lệ."""
        total = len(analyzed)
        if total == 0:
            return []

        if audit_limit is not None and audit_limit < total:
            indices = list(range(total))
            indices.sort(
                key=lambda i: (analyzed[i].needs_audit, analyzed[i].likes),
                reverse=True,
            )
            target_indices = sorted(indices[:audit_limit])
        else:
            target_indices = list(range(total))

        context_block = _format_video_context(media_result)
        raw_results: dict[int, dict[str, Any]] = {}

        batch_starts = range(0, len(target_indices), self.batch_size)
        iterator = (
            tqdm(batch_starts, desc="Gemma 4 E2B thẩm định thực tế", unit="batch")
            if show_progress
            else batch_starts
        )

        for start in iterator:
            batch_ids = target_indices[start : start + self.batch_size]
            chunk_pairs = [(idx, analyzed[idx]) for idx in batch_ids]
            raw_results.update(self._audit_chunk(chunk_pairs, context_block))

        audits: list[GemmaAuditItem] = []
        for idx, item in enumerate(analyzed):
            res = raw_results.get(idx)
            if res is None:
                audits.append(
                    GemmaAuditItem(
                        comment_index=idx,
                        gemma_audited=False,
                        gemma_polarity="",
                        gemma_emotion="",
                        gemma_emotion_vi="",
                        gemma_is_sarcasm=False,
                        gemma_agrees_polarity=None,
                        gemma_agrees_emotion=None,
                        gemma_reason="",
                    )
                )
            else:
                g_pol = res["gemma_polarity"]
                g_emo = res["gemma_emotion"]
                audits.append(
                    GemmaAuditItem(
                        comment_index=idx,
                        gemma_audited=True,
                        gemma_polarity=g_pol,
                        gemma_emotion=g_emo,
                        gemma_emotion_vi=EMOTION_VI_MAP.get(g_emo, g_emo),
                        gemma_is_sarcasm=res["gemma_is_sarcasm"],
                        gemma_agrees_polarity=(g_pol == item.polarity_label),
                        gemma_agrees_emotion=(g_emo == item.emotion_label),
                        gemma_reason=res["gemma_reason"],
                    )
                )
        return audits

    def generate_overall_verdict(
        self,
        analyzed: Sequence[AnalyzedComment],
        audits: Sequence[GemmaAuditItem],
        model_comparison: list[dict[str, Any]],
        media_result: MediaAnalysisResult | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        audited_pairs = [(a, g) for a, g in zip(analyzed, audits) if g.gemma_audited]
        if not audited_pairs:
            return {
                "accuracy_verdict": "Chưa kiểm định",
                "best_model": "",
                "audited_count": 0,
                "sarcasm_detected_count": 0,
                "sarcasm_rate_pct": 0.0,
                "strengths": [],
                "weaknesses": [],
                "summary_vi": "Chưa chạy bước thẩm định bằng Gemma 4 E2B.",
            }

        sarcasm_cnt = sum(1 for _, g in audited_pairs if g.gemma_is_sarcasm)
        sarcasm_rate = round(sarcasm_cnt * 100.0 / len(audited_pairs), 2)

        mismatches = [
            {
                "comment": a.original_text,
                "language": a.language,
                "hybrid_pred": f"{a.polarity_label} ({a.emotion_vi})",
                "gemma_truth": f"{g.gemma_polarity} ({g.gemma_emotion_vi})",
                "sarcasm": g.gemma_is_sarcasm,
                "reason": g.gemma_reason,
            }
            for a, g in audited_pairs
            if not g.gemma_agrees_polarity
        ][:12]

        comp_summary = [
            {
                "model": row["model"],
                "agreement_pct": row.get("agreement_vs_gemma_pct"),
                "macro_f1": row.get("macro_f1_vs_gemma"),
                "pos_pct": row["pos_pct"],
                "neg_pct": row["neg_pct"],
                "neu_pct": row["neu_pct"],
            }
            for row in model_comparison
        ]

        prompt = (
            f"BỐI CẢNH VIDEO:\n{_format_video_context(media_result)}\n\n"
            f"THỐNG KÊ THẨM ĐỊNH ({len(audited_pairs)} bình luận, {sarcasm_cnt} câu mỉa mai = {sarcasm_rate}%):\n"
            f"{json.dumps(comp_summary, ensure_ascii=False, indent=2)}\n\n"
            f"MỘT SỐ CÂU HỆ THỐNG HYBRID LỆCH SO VỚI GEMMA 4:\n"
            f"{json.dumps(mismatches, ensure_ascii=False, indent=2)}\n\n"
            "Hãy đưa ra đánh giá tổng quan trung thực xem hệ thống ML/DL đã đánh giá đúng chưa, "
            "mô hình nào khớp thực tế nhất, điểm mạnh và điểm yếu còn tồn tại."
        )

        try:
            verdict = self._call_ollama_json(
                [
                    {
                        "role": "system",
                        "content": "Bạn là chuyên gia đánh giá chất lượng hệ thống NLP song ngữ Việt – Anh. Trả về JSON súc tích, khách quan.",
                    },
                    {"role": "user", "content": prompt},
                ],
                VERDICT_SCHEMA,
            )
        except Exception as exc:
            best_row = max(
                [r for r in model_comparison if "Gemma 4" not in str(r.get("model", ""))],
                key=lambda r: float(r.get("agreement_vs_gemma_pct") or 0.0),
                default={},
            )
            verdict = {
                "accuracy_verdict": "Khá chính xác",
                "best_model": best_row.get("model", "Hybrid Ensemble"),
                "strengths": ["Hoàn thành phân loại đa mô hình song ngữ và đối chiếu thực tế."],
                "weaknesses": [f"Không tạo được nhận xét LLM chi tiết ({exc})."],
                "summary_vi": (
                    f"Đã thẩm định {len(audited_pairs)} bình luận, tỷ lệ mỉa mai {sarcasm_rate}%. "
                    f"Mô hình khớp cao nhất: {best_row.get('model', 'N/A')} "
                    f"({best_row.get('agreement_vs_gemma_pct', 'N/A')}%)."
                ),
            }

        verdict["audited_count"] = len(audited_pairs)
        verdict["sarcasm_detected_count"] = sarcasm_cnt
        verdict["sarcasm_rate_pct"] = sarcasm_rate
        return verdict


def build_model_comparison(
    analyzed: Sequence[AnalyzedComment],
    audits: Sequence[GemmaAuditItem] | None = None,
    timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Tạo bảng so sánh toàn diện các mô hình ML cổ điển, Lexicon, Deep Learning và Hybrid."""
    total = len(analyzed)
    timings = timings or {}
    if total == 0:
        return {"polarity_models": [], "emotion_models": []}

    has_vi = any(a.language == "vi" for a in analyzed)
    has_en = any(a.language == "en" for a in analyzed)
    if has_vi and has_en:
        dl_pol_name = "PhoBERT (VI) / Twitter-RoBERTa (EN)"
        dl_emo_name = "PhoBERT Emotion (VI) / DistilRoBERTa (EN)"
        dl_pol_time = round(
            float(timings.get("phobert_sentiment_sec") or 0.0)
            + float(timings.get("en_roberta_sentiment_sec") or 0.0),
            3,
        )
        dl_emo_time = round(
            float(timings.get("phobert_emotion_sec") or 0.0)
            + float(timings.get("en_roberta_emotion_sec") or 0.0),
            3,
        )
    elif has_en:
        dl_pol_name = "Twitter-RoBERTa Sentiment (EN)"
        dl_emo_name = "DistilRoBERTa Emotion (EN)"
        dl_pol_time = timings.get("en_roberta_sentiment_sec")
        dl_emo_time = timings.get("en_roberta_emotion_sec")
    else:
        dl_pol_name = "PhoBERT + underthesea (wonrax)"
        dl_emo_name = "PhoBERT Emotion (visolex/UIT-VSMEC)"
        dl_pol_time = timings.get("phobert_sentiment_sec")
        dl_emo_time = timings.get("phobert_emotion_sec")

    has_audit = bool(audits and any(g.gemma_audited for g in audits))
    audited_indices = [i for i, g in enumerate(audits) if g.gemma_audited] if audits else []
    gemma_pol_truth = [audits[i].gemma_polarity for i in audited_indices] if has_audit and audits else []
    gemma_emo_truth = [audits[i].gemma_emotion for i in audited_indices] if has_audit and audits else []

    polarity_specs: list[tuple[str, str, str, Any]] = [
        ("TF-IDF + Naive Bayes", "Classical ML", "pred_nb", timings.get("classical_ml_polarity_sec")),
        ("TF-IDF + Logistic Regression", "Classical ML", "pred_lr", timings.get("classical_ml_polarity_sec")),
        ("TF-IDF + Linear SVM", "Classical ML", "pred_svm", timings.get("classical_ml_polarity_sec")),
        ("Bilingual Lexicon + Rules", "Rule-based NLP", "pred_lexicon", timings.get("lexicon_rules_sec")),
    ]
    if has_vi and has_en:
        polarity_specs.extend(
            [
                ("PhoBERT (VI)", "Deep Learning", "pred_phobert", timings.get("phobert_sentiment_sec")),
                ("ViSoBERT (5CD-AI, VI)", "Deep Learning", "pred_visobert", timings.get("visobert_sec")),
                ("Twitter-RoBERTa (EN)", "Deep Learning", "pred_roberta", timings.get("en_roberta_sentiment_sec")),
            ]
        )
    elif has_en:
        polarity_specs.append((dl_pol_name, "Deep Learning", "pred_dl_primary", dl_pol_time))
    else:
        polarity_specs.extend(
            [
                ("ViSoBERT (5CD-AI, VI)", "Deep Learning", "pred_visobert", timings.get("visobert_sec")),
                (dl_pol_name, "Deep Learning", "pred_dl_primary", dl_pol_time),
            ]
        )
    polarity_specs.append(("Hybrid Ensemble (Hệ thống chính)", "Hybrid ML + DL", "polarity_label", None))

    polarity_rows: list[dict[str, Any]] = []
    for display_name, family, attr, t_sec in polarity_specs:
        # Với các mô hình theo từng ngôn ngữ, chỉ đánh giá trên đúng tập bình luận của ngôn ngữ đó
        if attr == "pred_visobert":
            preds_all = [a.pred_visobert for a in analyzed]
            eval_indices = [i for i, p in enumerate(preds_all) if p]
        elif attr == "pred_phobert":
            preds_all = [
                getattr(a, "pred_phobert", "") or (a.pred_dl_primary if a.language == "vi" else "")
                for a in analyzed
            ]
            eval_indices = [i for i, p in enumerate(preds_all) if p]
        elif attr == "pred_roberta":
            preds_all = [
                getattr(a, "pred_roberta", "") or (a.pred_dl_primary if a.language == "en" else "")
                for a in analyzed
            ]
            eval_indices = [i for i, p in enumerate(preds_all) if p]
        else:
            eval_indices = list(range(total))
            preds_all = [getattr(a, attr) for a in analyzed]

        sub_total = len(eval_indices) or 1
        pos_cnt = sum(1 for i in eval_indices if preds_all[i] == "POS")
        neu_cnt = sum(1 for i in eval_indices if preds_all[i] == "NEU")
        neg_cnt = sum(1 for i in eval_indices if preds_all[i] == "NEG")

        agreement_pct: float | None = None
        macro_f1: float | None = None
        kappa: float | None = None

        if has_audit and audited_indices and audits:
            aud_eval = [i for i in audited_indices if i in set(eval_indices)]
            if aud_eval:
                y_true_sub = [audits[i].gemma_polarity for i in aud_eval]
                y_pred_sub = [preds_all[i] for i in aud_eval]
                matches = sum(1 for yt, yp in zip(y_true_sub, y_pred_sub) if yt == yp)
                agreement_pct = round(matches * 100.0 / len(aud_eval), 2)
                macro_f1 = round(
                    float(
                        f1_score(
                            y_true_sub,
                            y_pred_sub,
                            labels=["POS", "NEG", "NEU"],
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    4,
                )
                try:
                    kappa_val = cohen_kappa_score(
                        y_true_sub, y_pred_sub, labels=["POS", "NEG", "NEU"]
                    )
                    kappa = 0.0 if np.isnan(kappa_val) else round(float(kappa_val), 4)
                except Exception:
                    kappa = None

        polarity_rows.append(
            {
                "model": display_name,
                "family": family,
                "total_comments": len(eval_indices),
                "pos_count": pos_cnt,
                "neu_count": neu_cnt,
                "neg_count": neg_cnt,
                "pos_pct": round(pos_cnt * 100.0 / sub_total, 2),
                "neu_pct": round(neu_cnt * 100.0 / sub_total, 2),
                "neg_pct": round(neg_cnt * 100.0 / sub_total, 2),
                "agreement_vs_gemma_pct": agreement_pct,
                "macro_f1_vs_gemma": macro_f1,
                "cohen_kappa_vs_gemma": kappa,
                "inference_time_sec": t_sec,
            }
        )

    if has_audit and audited_indices:
        g_pos = sum(1 for p in gemma_pol_truth if p == "POS")
        g_neu = sum(1 for p in gemma_pol_truth if p == "NEU")
        g_neg = sum(1 for p in gemma_pol_truth if p == "NEG")
        n_aud = len(audited_indices)
        polarity_rows.append(
            {
                "model": "Gemma 4 E2B (Thẩm định ngữ cảnh)",
                "family": "Multimodal LLM Reference",
                "total_comments": n_aud,
                "pos_count": g_pos,
                "neu_count": g_neu,
                "neg_count": g_neg,
                "pos_pct": round(g_pos * 100.0 / n_aud, 2),
                "neu_pct": round(g_neu * 100.0 / n_aud, 2),
                "neg_pct": round(g_neg * 100.0 / n_aud, 2),
                "agreement_vs_gemma_pct": 100.0,
                "macro_f1_vs_gemma": 1.0,
                "cohen_kappa_vs_gemma": 1.0,
                "inference_time_sec": timings.get("gemma_audit_sec"),
            }
        )

    emotion_specs = [
        ("TF-IDF + Logistic Regression (Emotion)", "Classical ML", "emotion_ml_label", timings.get("classical_ml_emotion_sec")),
        (dl_emo_name, "Deep Learning", "emotion_label", dl_emo_time),
    ]
    emotion_rows: list[dict[str, Any]] = []
    for display_name, family, attr, t_sec in emotion_specs:
        preds = [getattr(a, attr) for a in analyzed]
        agreement_pct = None
        macro_f1 = None
        if has_audit and audited_indices:
            sub_preds = [preds[i] for i in audited_indices]
            matches = sum(1 for yt, yp in zip(gemma_emo_truth, sub_preds) if yt == yp)
            agreement_pct = round(matches * 100.0 / len(audited_indices), 2)
            macro_f1 = round(
                float(
                    f1_score(
                        gemma_emo_truth,
                        sub_preds,
                        labels=list(EMOTION_LABELS),
                        average="macro",
                        zero_division=0,
                    )
                ),
                4,
            )
        emotion_rows.append(
            {
                "model": display_name,
                "family": family,
                "agreement_vs_gemma_pct": agreement_pct,
                "macro_f1_vs_gemma": macro_f1,
                "inference_time_sec": t_sec,
            }
        )

    return {
        "polarity_models": polarity_rows,
        "emotion_models": emotion_rows,
    }
