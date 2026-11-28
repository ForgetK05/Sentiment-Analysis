from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import io
import json
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError


def _ensure_windows_proactor_loop() -> None:
    """Đảm bảo thread hiện tại trên Windows dùng ProactorEventLoop để Playwright mở subprocess."""
    if sys.platform != "win32":
        return
    policy_cls = getattr(asyncio, "WindowsProactorEventLoopPolicy", None)
    if policy_cls is not None and not isinstance(asyncio.get_event_loop_policy(), policy_cls):
        asyncio.set_event_loop_policy(policy_cls())
    try:
        loop = asyncio.get_event_loop()
        if not isinstance(loop, asyncio.ProactorEventLoop):
            if not loop.is_running():
                loop.close()
            asyncio.set_event_loop(asyncio.new_event_loop())
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


COMMENT_COLUMN_CANDIDATES = (
    "comment",
    "original_text",
    "nội dung bình luận",
    "text",
    "content",
    "noi_dung",
    "binh_luan",
    "message",
)

_COMMENT_CSV_COLUMNS: tuple[str, ...] = (
    "comment",
    "comment_id",
    "is_reply",
    "parent_comment_id",
    "reply_to_comment_id",
    "reply_to_username",
    "username",
    "nickname",
    "likes",
    "reply_count",
    "created_at",
    "post_url",
)


def _clean_id(val: Any) -> str:
    """Chuẩn hóa ID TikTok dạng chuỗi (trả về '' nếu là None, '0', 'nan')."""
    if val is None:
        return ""
    s = str(val).strip()
    if not s or s == "0" or s.lower() == "nan":
        return ""
    return s


def _parse_bool(val: Any) -> bool:
    """Chuyển đổi an toàn giá trị bool/chuỗi/số sang bool thực thụ."""
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    if isinstance(val, (int, float)):
        try:
            if pd.isna(val):
                return False
        except Exception:
            pass
        return bool(val)
    s = str(val).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "", "no", "n", "nan", "none"}:
        return False
    return False


def _make_comment_adder(
    collected: dict[str, dict[str, Any]],
) -> Callable[[list[dict[str, Any]]], int]:
    """Tạo hàm thêm bình luận có chống trùng và nâng cấp dòng DOM fallback bằng dữ liệu API."""
    seen_content_keys: set[str] = set()
    fallback_key_map: dict[str, str] = {}

    def add_rows(rows: list[dict[str, Any]]) -> int:
        added = 0
        for row in rows:
            cid = _clean_id(row.get("comment_id"))
            username = str(row.get("username") or "").strip().lower()
            comment_text = str(row.get("comment") or "").strip()
            content_key = f"{username}::{comment_text}"

            if cid:
                if cid in collected:
                    continue
                old_fb_key = fallback_key_map.pop(content_key, None)
                if old_fb_key and old_fb_key in collected:
                    collected.pop(old_fb_key, None)
                    collected[cid] = row
                    seen_content_keys.add(content_key)
                    continue
                collected[cid] = row
                seen_content_keys.add(content_key)
                added += 1
            else:
                if content_key in seen_content_keys:
                    continue
                fb_key = f"dom::{content_key}"
                if fb_key not in collected:
                    collected[fb_key] = row
                    seen_content_keys.add(content_key)
                    fallback_key_map[content_key] = fb_key
                    added += 1
        return added

    return add_rows


_HASHTAG_RE = re.compile(r"#([^\s#.,!?;:()\[\]{}\"']+)")
_DEFAULT_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.tiktok.com/",
}
_UNIVERSAL_DATA_RE = re.compile(
    r'<script[^>]+id=["\']__UNIVERSAL_DATA_FOR_REHYDRATION__["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)
_SIGI_STATE_RE = re.compile(
    r'<script[^>]+id=["\']SIGI_STATE["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class TikTokMediaBundle:
    """Thông tin media và metadata tải tự động từ link TikTok (lưu trong data/<post_id>/)."""

    url: str
    resolved_url: str
    post_id: str
    media_type: str  # "video" | "image"
    media_paths: list[str]
    caption: str
    hashtags: list[str]
    uploader: str
    title: str
    duration: float | None
    like_count: int
    comment_count: int
    metadata_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_post_id(url: str) -> str:
    """Trích xuất ID bài viết từ URL TikTok (/video/... hoặc /photo/...)."""
    match = re.search(r"/(?:video|photo)/(\d+)", str(url))
    if match:
        return match.group(1)
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def extract_hashtags(text: str) -> list[str]:
    """Trích xuất danh sách hashtag (giữ nguyên thứ tự, loại trùng lặp)."""
    seen: set[str] = set()
    tags: list[str] = []
    for match in _HASHTAG_RE.findall(text or ""):
        clean = match.strip()
        lower = clean.lower()
        if clean and lower not in seen:
            seen.add(lower)
            tags.append(f"#{clean}")
    return tags


def _resolve_short_url(url: str, timeout: int = 20) -> str:
    """Giải mã link rút gọn vt.tiktok.com / vm.tiktok.com nếu cần."""
    clean_url = url.strip()
    if "vt.tiktok.com" not in clean_url and "vm.tiktok.com" not in clean_url:
        return clean_url
    try:
        resp = requests.head(
            clean_url,
            allow_redirects=True,
            timeout=timeout,
            headers=_DEFAULT_BROWSER_HEADERS,
        )
        if resp.url and resp.url != clean_url:
            return resp.url
    except requests.RequestException:
        pass
    try:
        with requests.get(
            clean_url,
            allow_redirects=True,
            stream=True,
            timeout=timeout,
            headers=_DEFAULT_BROWSER_HEADERS,
        ) as resp:
            if resp.url:
                return resp.url
    except requests.RequestException:
        pass
    return clean_url


def _cleanup_empty_dir(path: Path) -> None:
    """Dọn thư mục rỗng nếu quá trình tải gặp lỗi."""
    try:
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    except OSError:
        pass


def _find_image_post_struct(obj: Any) -> dict[str, Any] | None:
    """Tìm đệ quy dict chứa `imagePost.images` trong JSON cấu trúc TikTok."""
    if isinstance(obj, dict):
        img_post = obj.get("imagePost")
        if isinstance(img_post, dict) and isinstance(img_post.get("images"), list) and img_post["images"]:
            return obj
        for val in obj.values():
            found = _find_image_post_struct(val)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_image_post_struct(item)
            if found is not None:
                return found
    return None


def _extract_item_struct_from_universal_data(data: dict[str, Any]) -> dict[str, Any] | None:
    """Trích `itemStruct` từ JSON `__UNIVERSAL_DATA_FOR_REHYDRATION__` hoặc API `/api/item/detail/`."""
    if not isinstance(data, dict):
        return None

    # 1. API /api/item/detail/ -> itemInfo.itemStruct
    item_info = data.get("itemInfo")
    if isinstance(item_info, dict) and isinstance(item_info.get("itemStruct"), dict):
        return item_info["itemStruct"]

    # 2. __UNIVERSAL_DATA_FOR_REHYDRATION__ -> __DEFAULT_SCOPE__
    default_scope = data.get("__DEFAULT_SCOPE__")
    if isinstance(default_scope, dict):
        for scope_key in ("webapp.video-detail", "webapp.reflow.video.detail"):
            detail = default_scope.get(scope_key)
            if isinstance(detail, dict):
                scope_info = detail.get("itemInfo")
                if isinstance(scope_info, dict) and isinstance(scope_info.get("itemStruct"), dict):
                    return scope_info["itemStruct"]

    # 3. SIGI_STATE -> ItemModule
    item_module = data.get("ItemModule")
    if isinstance(item_module, dict):
        for val in item_module.values():
            if isinstance(val, dict) and val.get("imagePost"):
                return val

    # 4. Quét đệ quy dự phòng
    return _find_image_post_struct(data)


def _extract_item_struct_from_html(html: str) -> dict[str, Any] | None:
    """Tách JSON nhúng trong HTML trang TikTok và tìm `itemStruct`."""
    if not html:
        return None
    for pattern in (_UNIVERSAL_DATA_RE, _SIGI_STATE_RE):
        match = pattern.search(html)
        if not match:
            continue
        raw_json = match.group(1).strip()
        if not raw_json:
            continue
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        found = _extract_item_struct_from_universal_data(parsed)
        if found is not None:
            return found
    return None


def _extract_url_list_from_image_entry(entry: Any) -> list[str]:
    """Lấy toàn bộ danh sách URL mirror cho 1 slide từ cấu trúc `imagePost.images[i]`."""
    if not isinstance(entry, dict):
        return []
    urls: list[str] = []
    for key in ("imageURL", "displayImage", "ownerWatermarkImage", "thumbnail"):
        node = entry.get(key)
        if isinstance(node, dict):
            url_list = node.get("urlList") or node.get("url_list")
            if isinstance(url_list, list):
                for u in url_list:
                    u_str = str(u).strip()
                    if u_str and u_str not in urls:
                        urls.append(u_str)
    for direct_key in ("url", "src"):
        val = entry.get(direct_key)
        if isinstance(val, str) and val.strip() and val.strip() not in urls:
            urls.append(val.strip())
    return urls


def _parse_photo_item_struct(item_struct: dict[str, Any]) -> dict[str, Any]:
    """Chuẩn hóa danh sách URL từng slide và metadata từ `itemStruct` của bài Photo Mode."""
    image_post = item_struct.get("imagePost") if isinstance(item_struct.get("imagePost"), dict) else {}
    raw_images = image_post.get("images") if isinstance(image_post.get("images"), list) else []

    image_url_groups: list[list[str]] = []
    for entry in raw_images:
        urls = _extract_url_list_from_image_entry(entry)
        if urls:
            image_url_groups.append(urls)

    author = item_struct.get("author") if isinstance(item_struct.get("author"), dict) else {}
    stats = item_struct.get("stats") if isinstance(item_struct.get("stats"), dict) else {}
    if not stats and isinstance(item_struct.get("statsV2"), dict):
        stats = item_struct["statsV2"]

    caption = str(item_struct.get("desc") or "").strip()
    title = str(image_post.get("title") or "").strip()
    if not caption and title:
        caption = title
    uploader = str(author.get("uniqueId") or author.get("nickname") or "").strip()

    def _safe_int(val: Any) -> int:
        try:
            return int(val or 0)
        except (TypeError, ValueError):
            return 0

    return {
        "post_id": str(item_struct.get("id") or "").strip(),
        "image_url_groups": image_url_groups,
        "caption": caption,
        "title": title,
        "uploader": uploader,
        "like_count": _safe_int(stats.get("diggCount")),
        "comment_count": _safe_int(stats.get("commentCount")),
    }


def _save_image_bytes_as_jpeg(
    raw_bytes: bytes,
    out_file: Path,
    seen_hashes: set[str],
    min_dim: int = 64,
) -> bool:
    """Kiểm tra ảnh hợp lệ, khử trùng lặp MD5 và lưu chuẩn JPEG (.jpg)."""
    if not raw_bytes:
        return False

    digest = hashlib.md5(raw_bytes).hexdigest()
    if digest in seen_hashes:
        return False

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.load()
            width, height = img.size
            if min(width, height) < min_dim:
                return False
            rgb = img.convert("RGB")
            out_file.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(out_file, format="JPEG", quality=92)
    except (UnidentifiedImageError, OSError, ValueError):
        return False

    seen_hashes.add(digest)
    return True


def _download_slide_images(
    image_url_groups: Sequence[Sequence[str]],
    images_dir: Path,
    fetcher: Callable[[str], bytes | None],
    max_images: int | None = None,
) -> list[str]:
    """Tải toàn bộ slide (hỗ trợ nhiều CDN mirror cho mỗi slide) và lưu thành slide_01.jpg, ..."""
    images_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    seen_hashes: set[str] = set()
    groups = list(image_url_groups) if max_images is None else list(image_url_groups)[:max_images]

    slide_num = 0
    for url_group in groups:
        for img_url in url_group:
            raw_bytes = fetcher(img_url)
            if not raw_bytes:
                continue
            out_file = images_dir / f"slide_{slide_num + 1:02d}.jpg"
            if _save_image_bytes_as_jpeg(raw_bytes, out_file, seen_hashes):
                slide_num += 1
                saved_paths.append(str(out_file))
                break

    if not saved_paths:
        _cleanup_empty_dir(images_dir)
    return saved_paths


def _try_fetch_photo_post_http(
    url: str,
    images_dir: Path,
    timeout: int = 20,
    max_images: int | None = None,
) -> dict[str, Any] | None:
    """Tầng 1 (Nhanh, không mở trình duyệt): đọc JSON __UNIVERSAL_DATA_FOR_REHYDRATION__ để tải ảnh."""
    session = requests.Session()
    session.headers.update(_DEFAULT_BROWSER_HEADERS)
    try:
        resp = session.get(url, allow_redirects=True, timeout=timeout)
        if resp.status_code != 200 or not resp.text:
            return None
        resolved_url = resp.url or url
        item_struct = _extract_item_struct_from_html(resp.text)
        if item_struct is None:
            return None
        parsed = _parse_photo_item_struct(item_struct)
        if not parsed["image_url_groups"]:
            return None

        def _fetch_bytes(img_url: str) -> bytes | None:
            try:
                r = session.get(
                    img_url,
                    timeout=timeout,
                    headers={"Referer": resolved_url},
                )
                if r.status_code == 200 and r.content:
                    return r.content
            except requests.RequestException:
                return None
            return None

        saved_paths = _download_slide_images(
            parsed["image_url_groups"],
            images_dir=images_dir,
            fetcher=_fetch_bytes,
            max_images=max_images,
        )
        if not saved_paths:
            return None

        return {
            "media_paths": saved_paths,
            "caption": parsed["caption"],
            "title": parsed["title"],
            "uploader": parsed["uploader"],
            "like_count": parsed["like_count"],
            "comment_count": parsed["comment_count"],
            "post_id": parsed["post_id"] or extract_post_id(resolved_url),
            "resolved_url": resolved_url,
        }
    except requests.RequestException:
        return None
    finally:
        session.close()


def _extract_dom_photo_urls(page: Any) -> list[str]:
    """Quét DOM tìm các ảnh slide thực tế (loại avatar, icon, thumbnail đề xuất)."""
    try:
        found = page.evaluate(
            """() => {
                const imgs = Array.from(document.querySelectorAll('img'));
                const candidates = [];
                for (const img of imgs) {
                    const src = (img.currentSrc || img.src || '').trim();
                    if (!src) continue;
                    const lower = src.toLowerCase();
                    if (lower.includes('avatar') || lower.includes('logo') || lower.includes('emoji')) continue;
                    if (img.closest('[data-e2e="comment-list"], [class*="DivCommentListContainer"], [data-e2e="search-card-list"]')) continue;
                    const rect = img.getBoundingClientRect();
                    const w = img.naturalWidth || rect.width || 0;
                    const h = img.naturalHeight || rect.height || 0;
                    if (w < 200 || h < 200) continue;
                    const isCdn = (
                        lower.includes('tiktokcdn')
                        || lower.includes('ibyteimg')
                        || lower.includes('byteoversea')
                        || lower.includes('muscdn')
                    );
                    if (!isCdn) continue;
                    candidates.push(src);
                }
                return Array.from(new Set(candidates));
            }"""
        )
        return [str(u).strip() for u in (found or []) if str(u).strip()]
    except Exception:
        return []


def _expand_visible_replies(
    page: Any,
    collected: dict[str, dict[str, Any]],
    add_rows: Callable[[list[dict[str, Any]]], int],
    get_resolved_url: Callable[[], str],
    max_comments: int,
    max_replies_per_comment: int | None = None,
    reply_pause_sec: float = 1.5,
) -> int:
    """Tự động bấm các nút 'Xem thêm câu trả lời / View replies' đang hiển thị trên DOM."""
    total_added = 0
    consecutive_empty = 0

    while len(collected) < max_comments:
        try:
            target_info = page.evaluate(
                r"""(maxPerComment) => {
                    const viewRe = /^(?:xem\s+(?:thêm\s+)?(?:\d+\s+)?(?:câu\s+trả\s+lời|phản\s+hồi)(?:\s+khác)?|xem\s+thêm(?:\s*\(\d+\))?|view\s+(?:\d+\s+)?(?:more\s+)?repl(?:y|ies)|view\s+more(?:\s*\(\d+\))?)$/i;
                    const hideRe = /^(?:ẩn|hide|thu\s*gọn|collapse|trả\s*lời|reply)(?:\b|\s|$)/i;

                    document.querySelectorAll('[data-agy-next-reply-btn]').forEach((el) => {
                        el.removeAttribute('data-agy-next-reply-btn');
                    });

                    const selector = [
                        '[data-e2e="view-more-1"]',
                        '[data-e2e="view-more-2"]',
                        '[class*="DivViewRepliesContainer"] p',
                        '[class*="DivViewRepliesContainer"] span',
                        '[class*="DivViewRepliesContainer"] button',
                        '[class*="DivViewMoreRepliesWrapper"] p',
                        '[class*="DivViewMoreRepliesWrapper"] span',
                        '[class*="DivViewMoreRepliesWrapper"] button',
                        '[class*="DivReplyContainer"] p',
                        '[class*="DivReplyContainer"] span',
                        '[class*="DivReplyContainer"] button',
                    ].join(', ');

                    const rawEls = Array.from(document.querySelectorAll(selector));
                    const matched = [];
                    for (const el of rawEls) {
                        const text = (el.innerText || '').replace(/\s+/g, ' ').trim();
                        if (!text || text.length > 45) continue;
                        if (hideRe.test(text) || !viewRe.test(text)) continue;
                        const style = window.getComputedStyle(el);
                        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;
                        const rect = el.getBoundingClientRect();
                        if (rect.width <= 0 || rect.height <= 0) continue;
                        matched.push(el);
                    }

                    // Ưu tiên node bên trong nhất để tránh bấm trùng container cha/con
                    const innermost = matched.filter(
                        (el) => !matched.some((other) => other !== el && el.contains(other))
                    );

                    for (const el of innermost) {
                        const container = el.closest('[class*="DivCommentObjectWrapper"], [class*="DivCommentItemContainer"]') || el.parentElement;
                        if (!container) continue;
                        const stalls = parseInt(container.dataset.agyReplyStalls || '0', 10);
                        if (stalls >= 2) continue;
                        const loaded = parseInt(container.dataset.agyRepliesLoaded || '0', 10);
                        if (maxPerComment !== null && loaded >= maxPerComment) continue;

                        if (!container.dataset.agyThreadId) {
                            window.__agyThreadSeq = (window.__agyThreadSeq || 0) + 1;
                            container.dataset.agyThreadId = String(window.__agyThreadSeq);
                        }
                        el.setAttribute('data-agy-next-reply-btn', '1');
                        return { found: true, threadId: container.dataset.agyThreadId };
                    }
                    return { found: false, threadId: '' };
                }""",
                max_replies_per_comment,
            )
        except Exception:
            break

        if not target_info or not target_info.get("found"):
            break

        thread_id = str(target_info.get("threadId") or "")
        before_count = len(collected)
        btn = page.locator('[data-agy-next-reply-btn="1"]').first

        clicked = False
        try:
            btn.scroll_into_view_if_needed(timeout=1500)
            btn.click(timeout=1500)
            clicked = True
        except Exception:
            try:
                btn.click(force=True, timeout=1500)
                clicked = True
            except Exception:
                clicked = False

        if clicked:
            deadline = time.monotonic() + max(0.3, reply_pause_sec)
            while time.monotonic() < deadline:
                if len(collected) > before_count:
                    time.sleep(0.25)
                    break
                time.sleep(0.15)
            if len(collected) == before_count:
                add_rows(_extract_dom_fallback(page, get_resolved_url(), include_replies=True))

        gained = len(collected) - before_count
        total_added += gained

        if thread_id:
            try:
                page.evaluate(
                    """({ threadId, gained }) => {
                        const container = document.querySelector(`[data-agy-thread-id="${threadId}"]`);
                        if (!container) return;
                        if (gained > 0) {
                            const prev = parseInt(container.dataset.agyRepliesLoaded || '0', 10);
                            container.dataset.agyRepliesLoaded = String(prev + gained);
                            container.dataset.agyReplyStalls = '0';
                        } else {
                            const prevStalls = parseInt(container.dataset.agyReplyStalls || '0', 10);
                            container.dataset.agyReplyStalls = String(prevStalls + 1);
                        }
                    }""",
                    {"threadId": thread_id, "gained": gained},
                )
            except Exception:
                pass

        if gained > 0:
            consecutive_empty = 0
        else:
            consecutive_empty += 1
            if consecutive_empty >= 6:
                break

    return total_added


def _scroll_and_collect_comments_on_page(
    page: Any,
    collected: dict[str, dict[str, Any]],
    add_rows: Callable[[list[dict[str, Any]]], int],
    get_resolved_url: Callable[[], str],
    max_comments: int,
    max_scrolls: int,
    scroll_pause_sec: float,
    include_replies: bool = True,
    max_replies_per_comment: int | None = None,
    reply_pause_sec: float = 1.5,
) -> None:
    """Cuộn danh sách bình luận và gom comment (kèm reply nếu bật) trên trang Playwright đang mở."""
    add_rows(_extract_dom_fallback(page, get_resolved_url(), include_replies=include_replies))
    if include_replies and len(collected) < max_comments:
        _expand_visible_replies(
            page=page,
            collected=collected,
            add_rows=add_rows,
            get_resolved_url=get_resolved_url,
            max_comments=max_comments,
            max_replies_per_comment=max_replies_per_comment,
            reply_pause_sec=reply_pause_sec,
        )

    idle_scrolls = 0
    for step in range(1, max_scrolls + 1):
        if len(collected) >= max_comments:
            print(f"Đã đạt giới hạn {max_comments} bình luận.")
            break

        before_count = len(collected)
        for comment_sel in (
            'div[data-e2e="comment-level-1"], div[data-e2e="comment-level-2"]',
            '[class*="DivCommentItemContainer"]',
            '[class*="DivCommentListContainer"] > div',
        ):
            try:
                locs = page.locator(comment_sel)
                cnt = locs.count()
                if cnt > 0:
                    locs.nth(cnt - 1).scroll_into_view_if_needed(timeout=1000)
                    break
            except Exception:
                pass

        try:
            page.evaluate(
                """() => {
                    const el = document.querySelector('[class*="DivCommentListContainer"], div[data-e2e="comment-level-1"]');
                    let node = el ? el.parentElement : null;
                    while (node && node !== document.body) {
                        const style = window.getComputedStyle(node);
                        const overflowY = style.overflowY;
                        if ((overflowY === 'auto' || overflowY === 'scroll') && node.scrollHeight > node.clientHeight + 20) {
                            node.scrollTop += 1400;
                            break;
                        }
                        node = node.parentElement;
                    }
                    window.scrollBy(0, 1400);
                }"""
            )
        except Exception:
            pass

        page.keyboard.press("PageDown")
        time.sleep(scroll_pause_sec)

        if len(collected) == before_count:
            add_rows(_extract_dom_fallback(page, get_resolved_url(), include_replies=include_replies))

        if include_replies and len(collected) < max_comments:
            _expand_visible_replies(
                page=page,
                collected=collected,
                add_rows=add_rows,
                get_resolved_url=get_resolved_url,
                max_comments=max_comments,
                max_replies_per_comment=max_replies_per_comment,
                reply_pause_sec=reply_pause_sec,
            )

        after_count = len(collected)
        if include_replies:
            reply_cnt = sum(1 for r in collected.values() if r.get("is_reply"))
            main_cnt = after_count - reply_cnt
            print(
                f"  Lần cuộn {step}/{max_scrolls}: thu được {after_count} bình luận "
                f"({main_cnt} chính, {reply_cnt} reply)"
            )
        else:
            print(f"  Lần cuộn {step}/{max_scrolls}: thu được {after_count} bình luận")

        if after_count == before_count:
            idle_scrolls += 1
            if idle_scrolls >= 4:
                print("Không tải thêm được bình luận mới sau 4 lần cuộn liên tiếp, dừng thu thập.")
                if after_count <= 20:
                    print(
                        "[Gợi ý] TikTok thường giới hạn chỉ hiện 10-20 bình luận đầu nếu chưa đăng nhập trên cửa sổ Edge đó."
                    )
                break
        else:
            idle_scrolls = 0


def _download_photo_post_playwright(
    url: str,
    images_dir: Path,
    browser_channel: str | None = "msedge",
    headless: bool = False,
    interactive_pause: bool = True,
    max_images: int | None = None,
    collect_comments: bool = False,
    max_comments: int = 200,
    max_scrolls: int = 40,
    scroll_pause_sec: float = 3.0,
    include_replies: bool = True,
    max_replies_per_comment: int | None = None,
) -> dict[str, Any]:
    """Tầng 2: Dùng Microsoft Edge tải toàn bộ ảnh Photo Mode (và thu thập comment trong cùng phiên nếu cần)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ImportError(
            "Bài viết dạng ảnh (/photo/...) cần `playwright` để tải. "
            "Hãy chạy: pip install playwright"
        ) from exc

    resolved_url = url
    captured_item_structs: list[dict[str, Any]] = []
    collected_comments: dict[str, dict[str, Any]] = {}
    saved_paths: list[str] = []
    photo_meta: dict[str, Any] = {
        "caption": "",
        "title": "",
        "uploader": "",
        "like_count": 0,
        "comment_count": 0,
        "post_id": "",
    }

    add_comment_rows = _make_comment_adder(collected_comments)

    def _run_sync() -> None:
        nonlocal resolved_url, saved_paths
        _ensure_windows_proactor_loop()
        with sync_playwright() as p:
            launch_kwargs: dict[str, Any] = {"headless": headless}
            if browser_channel:
                launch_kwargs["channel"] = browser_channel
            browser = p.chromium.launch(**launch_kwargs)
            context = browser.new_context(
                viewport={"width": 1280, "height": 860},
                locale="vi-VN",
            )
            page = context.new_page()

            def handle_response(response: Any) -> None:
                resp_url = response.url
                if response.status != 200:
                    return

                if "/api/item/detail/" in resp_url:
                    try:
                        payload = response.json()
                        struct = _extract_item_struct_from_universal_data(payload)
                        if struct is not None:
                            captured_item_structs.append(struct)
                    except Exception:
                        pass
                    return

                if not collect_comments:
                    return
                is_main = (
                    "/api/comment/list/" in resp_url
                    and "/api/comment/list/reply/" not in resp_url
                )
                is_reply = "/api/comment/list/reply/" in resp_url
                if not (is_main or (include_replies and is_reply)):
                    return
                try:
                    payload = response.json()
                    add_comment_rows(
                        _parse_comment_items(
                            payload,
                            resolved_url,
                            is_reply_response=is_reply,
                            include_replies=include_replies,
                        )
                    )
                except Exception:
                    pass

            page.on("response", handle_response)

            print(f"Đang mở trình duyệt tới bài viết Photo Mode: {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            time.sleep(3.5)
            resolved_url = page.url

            if interactive_pause and not headless:
                print("\n[Tạm dừng] Trên cửa sổ Microsoft Edge vừa mở:")
                print("  1. Đóng popup đăng nhập (hoặc đăng nhập nếu TikTok yêu cầu).")
                print("  2. Đảm bảo ảnh bài viết (và phần bình luận nếu cần) đã hiển thị / hoàn tất xác minh.")
                input("Khi sẵn sàng, nhấn Enter tại đây để tiếp tục tải ảnh & thu thập...")

            resolved_url = page.url

            # Ưu tiên 1: gói /api/item/detail/
            image_url_groups: list[list[str]] = []
            for struct in captured_item_structs:
                parsed = _parse_photo_item_struct(struct)
                if parsed["image_url_groups"]:
                    image_url_groups = parsed["image_url_groups"]
                    for k in ("caption", "title", "uploader", "like_count", "comment_count", "post_id"):
                        if parsed[k]:
                            photo_meta[k] = parsed[k]
                    break

            # Ưu tiên 2: JSON nhúng trong DOM trang đã render
            if not image_url_groups:
                try:
                    html = page.content()
                    struct = _extract_item_struct_from_html(html)
                    if struct is not None:
                        parsed = _parse_photo_item_struct(struct)
                        if parsed["image_url_groups"]:
                            image_url_groups = parsed["image_url_groups"]
                            for k in ("caption", "title", "uploader", "like_count", "comment_count", "post_id"):
                                if parsed[k]:
                                    photo_meta[k] = parsed[k]
                except Exception:
                    pass

            # Ưu tiên 3: Quét DOM carousel (+ bấm chuyển slide nếu lazy-load)
            if not image_url_groups:
                dom_urls: list[str] = []
                for _ in range(12):
                    for u in _extract_dom_photo_urls(page):
                        if u not in dom_urls:
                            dom_urls.append(u)
                    next_btn = page.locator(
                        '[data-e2e="arrow-right"], button[aria-label*="Next"], [class*="SwiperButtonNext"]'
                    ).first
                    try:
                        if next_btn.is_visible(timeout=400):
                            next_btn.click(timeout=800)
                            time.sleep(0.5)
                        else:
                            break
                    except Exception:
                        break
                image_url_groups = [[u] for u in dom_urls]

            if not photo_meta["caption"]:
                try:
                    photo_meta["caption"] = (
                        page.evaluate(
                            """() => {
                                const desc = document.querySelector('[data-e2e="browse-video-desc"], [data-e2e="video-desc"]');
                                return desc ? desc.innerText.trim() : '';
                            }"""
                        )
                        or ""
                    )
                except Exception:
                    pass

            def _fetch_bytes_with_context(img_url: str) -> bytes | None:
                try:
                    api_resp = context.request.get(
                        img_url,
                        timeout=20000,
                        headers={"Referer": resolved_url or "https://www.tiktok.com/"},
                    )
                    if api_resp.ok:
                        body = api_resp.body()
                        if body:
                            return body
                except Exception:
                    pass
                try:
                    r = requests.get(
                        img_url,
                        timeout=20,
                        headers={**_DEFAULT_BROWSER_HEADERS, "Referer": resolved_url or "https://www.tiktok.com/"},
                    )
                    if r.status_code == 200 and r.content:
                        return r.content
                except requests.RequestException:
                    pass
                return None

            saved_paths = _download_slide_images(
                image_url_groups,
                images_dir=images_dir,
                fetcher=_fetch_bytes_with_context,
                max_images=max_images,
            )

            if collect_comments:
                _scroll_and_collect_comments_on_page(
                    page=page,
                    collected=collected_comments,
                    add_rows=add_comment_rows,
                    get_resolved_url=lambda: resolved_url,
                    max_comments=max_comments,
                    max_scrolls=max_scrolls,
                    scroll_pause_sec=scroll_pause_sec,
                    include_replies=include_replies,
                    max_replies_per_comment=max_replies_per_comment,
                )

            browser.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(_run_sync).result()

    if not saved_paths:
        _cleanup_empty_dir(images_dir)
        raise RuntimeError(
            "Không thể tự động tải ảnh từ bài viết Photo Mode này (có thể TikTok yêu cầu đăng nhập/xác minh). "
            "Vui lòng tải ảnh/video thủ công."
        )

    return {
        "media_paths": saved_paths,
        "caption": photo_meta["caption"],
        "title": photo_meta["title"],
        "uploader": photo_meta["uploader"],
        "like_count": photo_meta["like_count"],
        "comment_count": photo_meta["comment_count"],
        "post_id": photo_meta["post_id"] or extract_post_id(resolved_url),
        "resolved_url": resolved_url,
        "comment_rows": list(collected_comments.values())[:max_comments] if collect_comments else [],
    }


def _build_ytdlp_options(outtmpl: str, has_ffmpeg: bool | None = None) -> dict[str, Any]:
    """Tạo cấu hình yt-dlp an toàn: tự chọn định dạng gộp sẵn nếu máy chưa cài ffmpeg."""
    ffmpeg_ok = shutil.which("ffmpeg") is not None if has_ffmpeg is None else bool(has_ffmpeg)
    opts: dict[str, Any] = {
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    if ffmpeg_ok:
        opts["format"] = "bv*+ba/b[ext=mp4]/b"
        opts["merge_output_format"] = "mp4"
    else:
        opts["format"] = "b[ext=mp4]/best[ext=mp4]/b"
    return opts


def _build_photo_bundle(
    url: str,
    post_data_dir: Path,
    photo_result: dict[str, Any],
) -> TikTokMediaBundle:
    """Đóng gói TikTokMediaBundle cho bài Photo Mode và ghi metadata.json."""
    actual_id = str(photo_result.get("post_id") or post_data_dir.name)
    caption = str(photo_result.get("caption") or "").strip()
    bundle = TikTokMediaBundle(
        url=url,
        resolved_url=str(photo_result.get("resolved_url") or url),
        post_id=actual_id,
        media_type="image",
        media_paths=list(photo_result["media_paths"]),
        caption=caption,
        hashtags=extract_hashtags(caption),
        uploader=str(photo_result.get("uploader") or "").strip(),
        title=str(photo_result.get("title") or "").strip(),
        duration=None,
        like_count=int(photo_result.get("like_count") or 0),
        comment_count=int(photo_result.get("comment_count") or 0),
    )
    meta_path = post_data_dir / "metadata.json"
    bundle.metadata_path = str(meta_path)
    meta_path.write_text(
        json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return bundle


def _write_comments_csv(rows: list[dict[str, Any]], out_path: Path) -> str:
    """Xuất danh sách bình luận ra file CSV chuẩn utf-8-sig."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=list(_COMMENT_CSV_COLUMNS))
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return str(out_path)


def _download_video_with_ytdlp(
    url: str,
    resolved_url: str,
    post_id: str,
    post_data_dir: Path,
) -> TikTokMediaBundle:
    """Tải video MP4 và metadata bằng yt-dlp."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise ImportError("Chưa cài thư viện `yt-dlp`. Hãy chạy: pip install yt-dlp") from exc

    ydl_opts = _build_ytdlp_options(str(post_data_dir / "video.%(ext)s"))
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(resolved_url, download=True)
        downloaded_path = Path(ydl.prepare_filename(info))
        mp4_candidate = post_data_dir / "video.mp4"
        if mp4_candidate.is_file():
            downloaded_path = mp4_candidate
        if not downloaded_path.is_file():
            matches = sorted(post_data_dir.glob("video.*"))
            video_matches = [
                m for m in matches if m.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}
            ]
            if video_matches:
                downloaded_path = video_matches[0]

        if not downloaded_path.is_file():
            raise FileNotFoundError(
                f"yt-dlp không tạo được file video trong {post_data_dir}"
            )

        caption = str(info.get("description") or info.get("title") or "").strip()
        hashtags = extract_hashtags(caption)
        if not hashtags and isinstance(info.get("tags"), list):
            hashtags = [f"#{str(t).lstrip('#')}" for t in info["tags"] if str(t).strip()]

        actual_id = str(info.get("id") or post_id)
        bundle = TikTokMediaBundle(
            url=url,
            resolved_url=str(info.get("webpage_url") or resolved_url),
            post_id=actual_id,
            media_type="video",
            media_paths=[str(downloaded_path)],
            caption=caption,
            hashtags=hashtags,
            uploader=str(info.get("uploader") or info.get("creator") or "").strip(),
            title=str(info.get("title") or "").strip(),
            duration=float(info["duration"]) if info.get("duration") is not None else None,
            like_count=int(info.get("like_count") or 0),
            comment_count=int(info.get("comment_count") or 0),
        )
        meta_path = post_data_dir / "metadata.json"
        bundle.metadata_path = str(meta_path)
        meta_path.write_text(
            json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return bundle


def download_tiktok_media(
    url: str,
    data_dir: str | Path = "data",
    browser_channel: str | None = "msedge",
    headless: bool = False,
    interactive_pause: bool = True,
    max_images: int | None = None,
) -> TikTokMediaBundle:
    """Tự động tải video (.mp4) hoặc bộ ảnh Photo Mode cùng metadata vào data/<post_id>/."""
    bundle, _ = collect_tiktok_post(
        url=url,
        data_dir=data_dir,
        collect_comments=False,
        browser_channel=browser_channel,
        headless=headless,
        interactive_pause=interactive_pause,
        max_images=max_images,
    )
    return bundle


def collect_tiktok_post(
    url: str,
    data_dir: str | Path = "data",
    collect_comments: bool = True,
    max_comments: int = 200,
    max_scrolls: int = 40,
    scroll_pause_sec: float = 3.0,
    include_replies: bool = True,
    max_replies_per_comment: int | None = None,
    browser_channel: str | None = "msedge",
    headless: bool = False,
    interactive_pause: bool = True,
    max_images: int | None = None,
) -> tuple[TikTokMediaBundle, str | None]:
    """Tải media + thu thập comment trong một luồng thống nhất (gộp 1 cửa sổ Edge cho Photo Mode khi cần)."""
    resolved_url = _resolve_short_url(url)
    post_id = extract_post_id(resolved_url)
    post_data_dir = Path(data_dir) / post_id
    post_data_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = post_data_dir / "raw_comments.csv"

    is_photo_url = "/photo/" in resolved_url
    ydl_error: Exception | None = None

    if not is_photo_url:
        try:
            bundle = _download_video_with_ytdlp(
                url=url,
                resolved_url=resolved_url,
                post_id=post_id,
                post_data_dir=post_data_dir,
            )
            crawled_csv: str | None = None
            if collect_comments:
                print(f"[Chuẩn bị dữ liệu] Đang thu thập bình luận vào {raw_csv_path}...")
                _, crawled_csv = crawl_tiktok_comments(
                    url=resolved_url,
                    output_csv=raw_csv_path,
                    data_dir=data_dir,
                    max_comments=max_comments,
                    max_scrolls=max_scrolls,
                    scroll_pause_sec=scroll_pause_sec,
                    include_replies=include_replies,
                    max_replies_per_comment=max_replies_per_comment,
                    browser_channel=browser_channel,
                    headless=headless,
                    interactive_pause=interactive_pause,
                )
            return bundle, crawled_csv
        except Exception as exc:
            ydl_error = exc

    # Tầng 1 HTTP cho Photo Mode: nếu lấy được ảnh luôn thì chỉ cần mở Edge 1 lần cho comment
    http_photo = _try_fetch_photo_post_http(
        resolved_url,
        images_dir=post_data_dir / "images",
        max_images=max_images,
    )
    if http_photo is not None:
        bundle = _build_photo_bundle(url, post_data_dir, http_photo)
        crawled_csv = None
        if collect_comments:
            print(f"[Chuẩn bị dữ liệu] Đang thu thập bình luận vào {raw_csv_path}...")
            _, crawled_csv = crawl_tiktok_comments(
                url=bundle.resolved_url,
                output_csv=raw_csv_path,
                data_dir=data_dir,
                max_comments=max_comments,
                max_scrolls=max_scrolls,
                scroll_pause_sec=scroll_pause_sec,
                include_replies=include_replies,
                max_replies_per_comment=max_replies_per_comment,
                browser_channel=browser_channel,
                headless=headless,
                interactive_pause=interactive_pause,
            )
        return bundle, crawled_csv

    # Tầng 2 Playwright: gộp tải ảnh Photo Mode + thu thập bình luận trong CÙNG 1 cửa sổ Edge
    try:
        pw_photo = _download_photo_post_playwright(
            resolved_url,
            images_dir=post_data_dir / "images",
            browser_channel=browser_channel,
            headless=headless,
            interactive_pause=interactive_pause,
            max_images=max_images,
            collect_comments=collect_comments,
            max_comments=max_comments,
            max_scrolls=max_scrolls,
            scroll_pause_sec=scroll_pause_sec,
            include_replies=include_replies,
            max_replies_per_comment=max_replies_per_comment,
        )
    except Exception as pw_exc:
        _cleanup_empty_dir(post_data_dir)
        if not is_photo_url and ydl_error is not None:
            raise RuntimeError(
                f"Không thể tự động tải media từ link TikTok `{url}` (yt-dlp: {ydl_error}; Photo Mode: {pw_exc}). "
                "Hãy kiểm tra kết nối mạng, cập nhật `yt-dlp -U`, hoặc dùng thư mục cục bộ qua `--input-dir`."
            ) from pw_exc
        raise

    bundle = _build_photo_bundle(url, post_data_dir, pw_photo)
    crawled_csv = None
    if collect_comments:
        crawled_csv = _write_comments_csv(pw_photo.get("comment_rows") or [], raw_csv_path)
    return bundle, crawled_csv


def _format_unix_ts(ts_value: Any) -> str:
    if not ts_value:
        return ""
    try:
        dt = datetime.fromtimestamp(int(ts_value), tz=timezone.utc).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, OSError):
        return ""


def _build_comment_row(
    item: dict[str, Any],
    post_url: str,
    *,
    force_is_reply: bool = False,
    fallback_parent_id: str = "",
) -> dict[str, Any] | None:
    """Chuẩn hóa một object comment/reply TikTok thành dict đầy đủ metadata."""
    text = str(item.get("text") or "").strip()
    if not text:
        return None
    user = item.get("user") if isinstance(item.get("user"), dict) else {}
    cid = _clean_id(item.get("cid"))
    reply_id = _clean_id(item.get("reply_id"))
    reply_to_reply_id = _clean_id(item.get("reply_to_reply_id"))
    parent_id = reply_id or fallback_parent_id
    is_reply = bool(force_is_reply or parent_id)
    reply_to_username = (
        str(item.get("reply_to_username") or item.get("reply_to_nickname") or "").strip()
        if is_reply
        else ""
    )

    return {
        "comment": text,
        "comment_id": cid,
        "is_reply": is_reply,
        "parent_comment_id": parent_id if is_reply else "",
        "reply_to_comment_id": reply_to_reply_id if is_reply else "",
        "reply_to_username": reply_to_username,
        "username": str(user.get("unique_id") or "").strip(),
        "nickname": str(user.get("nickname") or "").strip(),
        "likes": int(item.get("digg_count") or 0),
        "reply_count": int(item.get("reply_comment_total") or 0),
        "created_at": _format_unix_ts(item.get("create_time")),
        "post_url": post_url,
    }


def _parse_comment_items(
    payload: dict[str, Any],
    post_url: str,
    *,
    is_reply_response: bool = False,
    include_replies: bool = True,
) -> list[dict[str, Any]]:
    """Chuyển JSON response của API bình luận TikTok thành danh sách dict chuẩn (gồm cả reply)."""
    if not isinstance(payload, dict):
        return []
    raw_comments = payload.get("comments") or []
    if not isinstance(raw_comments, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in raw_comments:
        if not isinstance(item, dict):
            continue
        row = _build_comment_row(item, post_url, force_is_reply=is_reply_response)
        if row is not None:
            rows.append(row)

        if include_replies and not is_reply_response:
            embedded_replies = item.get("reply_comment")
            if isinstance(embedded_replies, list):
                parent_cid = row["comment_id"] if row is not None else _clean_id(item.get("cid"))
                for rep in embedded_replies:
                    if not isinstance(rep, dict):
                        continue
                    rep_row = _build_comment_row(
                        rep,
                        post_url,
                        force_is_reply=True,
                        fallback_parent_id=parent_cid,
                    )
                    if rep_row is not None:
                        rows.append(rep_row)
    return rows


def _extract_dom_fallback(
    page: Any,
    post_url: str,
    include_replies: bool = False,
) -> list[dict[str, Any]]:
    """Dự phòng đọc bình luận đã render sẵn trên DOM nếu API response xảy ra trước khi gắn listener."""
    try:
        dom_items = page.evaluate(
            r"""(includeReplies) => {
                const results = [];
                const pickItems = (nodes, isReply) => {
                    for (const el of nodes) {
                        const textNode = isReply
                            ? (el.querySelector('span[data-e2e="comment-text"], p[data-e2e="comment-level-2"], span') || el)
                            : (el.querySelector('span[data-e2e="comment-text"], p[data-e2e="comment-level-1"], span') || el);
                        const text = (textNode.innerText || el.innerText || '').trim();
                        if (!text) continue;
                        const container = el.closest('[class*="DivCommentItemContainer"], [class*="DivReplyContainer"], [class*="DivCommentObjectWrapper"]') || el.parentElement;
                        let username = '';
                        if (container) {
                            const userSel = isReply
                                ? '[data-e2e="comment-username-2"], [data-e2e="comment-username"]'
                                : '[data-e2e="comment-username-1"], [data-e2e="comment-username"]';
                            const uEl = container.querySelector(userSel);
                            if (uEl) username = (uEl.innerText || '').trim();
                        }
                        results.push({ comment: text, username, is_reply: isReply });
                    }
                };

                let lvl1 = Array.from(document.querySelectorAll('div[data-e2e="comment-level-1"]'));
                if (!lvl1.length) {
                    lvl1 = Array.from(document.querySelectorAll('[class*="DivCommentItemContainer"]'));
                }
                pickItems(lvl1, false);

                if (includeReplies) {
                    let lvl2 = Array.from(document.querySelectorAll('div[data-e2e="comment-level-2"], p[data-e2e="comment-level-2"]'));
                    if (!lvl2.length) {
                        lvl2 = Array.from(document.querySelectorAll('[class*="DivReplyContainer"] [data-e2e="comment-text"]'));
                    }
                    pickItems(lvl2, true);
                }
                return results;
            }""",
            include_replies,
        )
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for item in dom_items or []:
        text = str(item.get("comment") or "").strip()
        if not text:
            continue
        rows.append(
            {
                "comment": text,
                "comment_id": "",
                "is_reply": bool(item.get("is_reply", False)),
                "parent_comment_id": "",
                "reply_to_comment_id": "",
                "reply_to_username": "",
                "username": str(item.get("username") or "").strip(),
                "nickname": "",
                "likes": 0,
                "reply_count": 0,
                "created_at": "",
                "post_url": post_url,
            }
        )
    return rows


def _collect_comments_sync(
    url: str,
    max_comments: int = 200,                  # Giới hạn số bình luận thu thập tối đa
    max_scrolls: int = 40,                    # Giới hạn số lần cuộn trang để tải thêm bình luận
    scroll_pause_sec: float = 3.0,            # Thời gian chờ sau mỗi lần cuộn (giây)
    include_replies: bool = True,             # Có thu thập bình luận trả lời (reply) hay không
    max_replies_per_comment: int | None = None,  # Giới hạn reply mỗi bình luận chính (None = lấy hết tới max_comments)
    browser_channel: str | None = "msedge",   # Trình duyệt Playwright channel (msedge, chrome, chromium, firefox, webkit)
    headless: bool = False,                   # Chạy trình duyệt ở chế độ không có giao diện
    interactive_pause: bool = True,           # Tạm dừng để người dùng xác minh captcha hoặc đăng nhập nếu cần
) -> tuple[list[dict[str, Any]], str]:
    """Chạy Playwright Sync API để thu thập comment TikTok."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ImportError(
            "Chưa cài đặt `playwright`. Hãy chạy: pip install playwright"
        ) from exc

    _ensure_windows_proactor_loop()
    collected: dict[str, dict[str, Any]] = {}
    resolved_url = url
    add_rows = _make_comment_adder(collected)

    with sync_playwright() as p:
        launch_kwargs: dict[str, Any] = {"headless": headless}
        if browser_channel:
            launch_kwargs["channel"] = browser_channel
        browser = p.chromium.launch(**launch_kwargs)
        context = browser.new_context(
            viewport={"width": 1280, "height": 860},
            locale="vi-VN",
        )
        page = context.new_page()

        def handle_response(response: Any) -> None:
            resp_url = response.url
            is_main = (
                "/api/comment/list/" in resp_url
                and "/api/comment/list/reply/" not in resp_url
            )
            is_reply = "/api/comment/list/reply/" in resp_url
            if not (is_main or (include_replies and is_reply)):
                return
            if response.status != 200:
                return
            try:
                payload = response.json()
                add_rows(
                    _parse_comment_items(
                        payload,
                        resolved_url,
                        is_reply_response=is_reply,
                        include_replies=include_replies,
                    )
                )
            except Exception:
                pass

        page.on("response", handle_response)

        print(f"Đang mở trình duyệt tới: {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(3.5)
        resolved_url = page.url

        if interactive_pause and not headless:
            print("\n[Tạm dừng] Trên cửa sổ trình duyệt vừa mở:")
            print("  1. Đóng popup đăng nhập (hoặc đăng nhập nếu TikTok yêu cầu).")
            print("  2. Mở phần bình luận (nếu chưa hiện) hoặc hoàn tất xác minh (nếu có).")
            input("Khi khu vực bình luận đã sẵn sàng, nhấn Enter tại đây để thu thập...")

        resolved_url = page.url
        _scroll_and_collect_comments_on_page(
            page=page,
            collected=collected,
            add_rows=add_rows,
            get_resolved_url=lambda: resolved_url,
            max_comments=max_comments,
            max_scrolls=max_scrolls,
            scroll_pause_sec=scroll_pause_sec,
            include_replies=include_replies,
            max_replies_per_comment=max_replies_per_comment,
        )

        browser.close()

    rows = list(collected.values())[:max_comments]
    return rows, resolved_url


def crawl_tiktok_comments(
    url: str,
    output_csv: str | Path | None = None,
    data_dir: str | Path = "data",
    max_comments: int = 200,
    max_scrolls: int = 40,
    scroll_pause_sec: float = 3.0,
    include_replies: bool = True,
    max_replies_per_comment: int | None = None,
    browser_channel: str | None = "msedge",
    headless: bool = False,
    interactive_pause: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    """Thu thập bình luận TikTok và lưu ra data/<post_id>/raw_comments.csv (utf-8-sig)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        rows, resolved_url = executor.submit(
            _collect_comments_sync,
            url,
            max_comments,
            max_scrolls,
            scroll_pause_sec,
            include_replies,
            max_replies_per_comment,
            browser_channel,
            headless,
            interactive_pause,
        ).result()

    post_id = extract_post_id(resolved_url)
    out_path = (
        Path(output_csv)
        if output_csv is not None
        else Path(data_dir) / post_id / "raw_comments.csv"
    )
    saved_csv = _write_comments_csv(rows, out_path)
    return rows, saved_csv


def _normalize_comments_df(df: pd.DataFrame, comment_col: str) -> pd.DataFrame:
    """Chuẩn hóa DataFrame bình luận và giữ lại các cột metadata quan trọng."""
    out = df.copy()
    out["comment"] = out[comment_col].fillna("").astype(str).str.strip()
    out = out[out["comment"] != ""].reset_index(drop=True)

    defaults: dict[str, Any] = {
        "comment_id": "",
        "is_reply": False,
        "parent_comment_id": "",
        "reply_to_comment_id": "",
        "reply_to_username": "",
        "username": "",
        "nickname": "",
        "likes": 0,
        "reply_count": 0,
        "created_at": "",
        "post_url": "",
    }
    for col, default_val in defaults.items():
        if col not in out.columns:
            out[col] = default_val
        elif col == "is_reply":
            out[col] = out[col].map(_parse_bool)
        elif col in {"likes", "reply_count"}:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
        else:
            out[col] = out[col].fillna("").astype(str)

    ordered = list(_COMMENT_CSV_COLUMNS)
    extra_cols = [c for c in out.columns if c not in ordered and c != comment_col]
    return out[ordered + extra_cols]


_TABULAR_ID_DTYPES: dict[str, Any] = {
    "comment_id": str,
    "parent_comment_id": str,
    "reply_to_comment_id": str,
    "reply_to_username": str,
    "username": str,
    "nickname": str,
}


def _select_comment_column(
    columns: Sequence[Any],
    text_column: str | None,
    source_path: Path,
) -> str:
    """Chọn cột chứa nội dung bình luận từ DataFrame tabular (hỗ trợ tiêu đề 2 dòng)."""
    col_strs = [str(c) for c in columns]
    if text_column is not None:
        if text_column in col_strs:
            return text_column
        raise ValueError(
            f"Không tìm thấy cột `{text_column}` trong {source_path}. Các cột hiện có: {col_strs}"
        )

    lower_map = {c.lower(): c for c in col_strs}
    for candidate in COMMENT_COLUMN_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    return col_strs[0]


def load_comments_dataframe(
    comments_source: str | Path | Sequence[str],
    text_column: str | None = None,
) -> pd.DataFrame:
    """Đọc bình luận từ list Python hoặc file CSV / JSON, giữ nguyên metadata."""
    if not isinstance(comments_source, (str, Path)):
        rows = [
            {"comment": str(item).strip()}
            for item in comments_source
            if str(item).strip()
        ]
        if not rows:
            return _normalize_comments_df(pd.DataFrame(columns=["comment"]), "comment")
        return _normalize_comments_df(pd.DataFrame(rows), "comment")

    path = Path(comments_source)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file comment: {path}")

    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=_TABULAR_ID_DTYPES)
        if df.empty:
            return _normalize_comments_df(pd.DataFrame(columns=["comment"]), "comment")

        # Hỗ trợ file output có tiêu đề 2 dòng dạng "<col>\n(chú thích)"
        df.columns = [str(c).split("\n", 1)[0].strip() for c in df.columns]
        col = _select_comment_column(list(df.columns), text_column=text_column, source_path=path)
        return _normalize_comments_df(df, col)

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

        dict_rows: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, str):
                if item.strip():
                    dict_rows.append({"comment": item.strip()})
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
                    row_copy = dict(item)
                    row_copy["comment"] = str(item[key]).strip()
                    dict_rows.append(row_copy)

        if not dict_rows:
            return _normalize_comments_df(pd.DataFrame(columns=["comment"]), "comment")
        return _normalize_comments_df(pd.DataFrame(dict_rows), "comment")

    raise ValueError(
        f"Định dạng file comment chưa hỗ trợ: {path.suffix} "
        "(chỉ hỗ trợ .csv, .json)"
    )