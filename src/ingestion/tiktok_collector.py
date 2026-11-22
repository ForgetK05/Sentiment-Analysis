from __future__ import annotations

import concurrent.futures
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import requests

COMMENT_COLUMN_CANDIDATES = (
    "comment",
    "text",
    "content",
    "noi_dung",
    "binh_luan",
    "message",
)

_HASHTAG_RE = re.compile(r"#([^\s#.,!?;:()\[\]{}\"']+)")


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
            headers={"User-Agent": "Mozilla/5.0"},
        )
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


def _download_photo_post_playwright(
    url: str,
    images_dir: Path,
    browser_channel: str | None = "msedge",
) -> tuple[list[str], str, str]:
    """Dự phòng tải ảnh cho bài viết TikTok Photo Mode (/photo/...) bằng Playwright."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ImportError(
            "Bài viết dạng ảnh (/photo/...) cần `playwright` để tải. "
            "Hãy chạy: pip install playwright"
        ) from exc

    image_urls: list[str] = []
    caption = ""
    resolved_url = url

    def _run_sync() -> None:
        nonlocal resolved_url, caption
        with sync_playwright() as p:
            launch_kwargs: dict[str, Any] = {"headless": True}
            if browser_channel:
                launch_kwargs["channel"] = browser_channel
            browser = p.chromium.launch(**launch_kwargs)
            context = browser.new_context(
                viewport={"width": 1280, "height": 860},
                locale="vi-VN",
            )
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            time.sleep(3.5)
            resolved_url = page.url

            try:
                caption = page.evaluate(
                    """() => {
                        const desc = document.querySelector('[data-e2e="browse-video-desc"], [data-e2e="video-desc"]');
                        return desc ? desc.innerText.trim() : '';
                    }"""
                ) or ""
            except Exception:
                caption = ""

            try:
                found_imgs = page.evaluate(
                    """() => {
                        const imgs = Array.from(document.querySelectorAll('img'));
                        return imgs
                            .map(img => img.src || '')
                            .filter(src => src.includes('tiktokcdn') && !src.includes('avatar'));
                    }"""
                )
                for img_url in found_imgs or []:
                    if img_url not in image_urls:
                        image_urls.append(img_url)
            except Exception:
                pass
            browser.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(_run_sync).result()

    images_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    for idx, img_url in enumerate(image_urls[:12], start=1):
        try:
            resp = requests.get(img_url, timeout=20)
            if resp.status_code == 200 and resp.content:
                out_file = images_dir / f"slide_{idx:02d}.jpg"
                out_file.write_bytes(resp.content)
                saved_paths.append(str(out_file))
        except requests.RequestException:
            continue

    if not saved_paths:
        _cleanup_empty_dir(images_dir)
        raise RuntimeError(
            "Không thể tự động tải ảnh từ bài viết Photo Mode này (có thể TikTok yêu cầu xác minh). "
            "Vui lòng tải ảnh/video thủ công."
        )
    return saved_paths, caption, resolved_url


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


def download_tiktok_media(
    url: str,
    data_dir: str | Path = "data",
    browser_channel: str | None = "msedge",
) -> TikTokMediaBundle:
    """Tự động tải video (.mp4) và trích xuất caption, hashtags, metadata vào data/<post_id>/."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise ImportError("Chưa cài thư viện `yt-dlp`. Hãy chạy: pip install yt-dlp") from exc

    resolved_url = _resolve_short_url(url)
    post_id = extract_post_id(resolved_url)
    post_data_dir = Path(data_dir) / post_id
    post_data_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = _build_ytdlp_options(str(post_data_dir / "video.%(ext)s"))

    is_photo_url = "/photo/" in resolved_url
    ydl_error: Exception | None = None

    if not is_photo_url:
        try:
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
        except Exception as exc:
            ydl_error = exc

    if is_photo_url or "/photo/" in resolved_url:
        try:
            saved_imgs, cap, final_url = _download_photo_post_playwright(
                resolved_url,
                images_dir=post_data_dir / "images",
                browser_channel=browser_channel,
            )
        except Exception:
            _cleanup_empty_dir(post_data_dir)
            raise

        bundle = TikTokMediaBundle(
            url=url,
            resolved_url=final_url,
            post_id=extract_post_id(final_url),
            media_type="image",
            media_paths=saved_imgs,
            caption=cap,
            hashtags=extract_hashtags(cap),
            uploader="",
            title="",
            duration=None,
            like_count=0,
            comment_count=0,
        )
        meta_path = post_data_dir / "metadata.json"
        bundle.metadata_path = str(meta_path)
        meta_path.write_text(
            json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return bundle

    _cleanup_empty_dir(post_data_dir)
    raise RuntimeError(
        f"Không thể tự động tải video từ link TikTok `{url}`: {ydl_error}. "
        "Hãy kiểm tra kết nối mạng, cập nhật `yt-dlp -U`, hoặc truyền file video qua `--media`."
    ) from ydl_error


def _format_unix_ts(ts_value: Any) -> str:
    if not ts_value:
        return ""
    try:
        dt = datetime.fromtimestamp(int(ts_value), tz=timezone.utc).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, OSError):
        return ""


def _parse_comment_items(payload: dict[str, Any], post_url: str) -> list[dict[str, Any]]:
    """Chuyển JSON response của API bình luận TikTok thành danh sách dict chuẩn."""
    if not isinstance(payload, dict):
        return []
    raw_comments = payload.get("comments") or []
    if not isinstance(raw_comments, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in raw_comments:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        user = item.get("user") if isinstance(item.get("user"), dict) else {}
        cid = str(item.get("cid") or "").strip()
        rows.append(
            {
                "comment": text,
                "comment_id": cid,
                "username": str(user.get("unique_id") or "").strip(),
                "nickname": str(user.get("nickname") or "").strip(),
                "likes": int(item.get("digg_count") or 0),
                "reply_count": int(item.get("reply_comment_total") or 0),
                "created_at": _format_unix_ts(item.get("create_time")),
                "post_url": post_url,
            }
        )
    return rows


def _extract_dom_fallback(page: Any, post_url: str) -> list[dict[str, Any]]:
    """Dự phòng đọc bình luận đã render sẵn trên DOM nếu API response xảy ra trước khi gắn listener."""
    rows: list[dict[str, Any]] = []
    selectors = [
        'div[data-e2e="comment-level-1"]',
        '[class*="DivCommentItemContainer"]',
    ]
    elements: list[Any] = []
    for sel in selectors:
        found = page.locator(sel).all()
        if found:
            elements = found
            break

    for el in elements:
        try:
            text_loc = el.locator(
                '[data-e2e="comment-level-1"] span, span[data-e2e="comment-text"], p[data-e2e="comment-level-1"]'
            ).first
            text = text_loc.inner_text(timeout=500).strip()
        except Exception:
            try:
                text = el.inner_text(timeout=500).strip()
            except Exception:
                text = ""
        if not text:
            continue

        username = ""
        try:
            username = (
                el.locator('[data-e2e="comment-username-1"], span[data-e2e="comment-username"]')
                .first.inner_text(timeout=300)
                .strip()
            )
        except Exception:
            pass

        rows.append(
            {
                "comment": text,
                "comment_id": "",
                "username": username,
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
    include_replies: bool = False,            # Có thu thập bình luận trả lời (reply) hay không
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

    collected: dict[str, dict[str, Any]] = {}
    resolved_url = url

    def add_rows(rows: list[dict[str, Any]]) -> int:
        added = 0
        for row in rows:
            dedup_key = row["comment_id"] or f"{row['username']}::{row['comment']}"
            if dedup_key not in collected:
                collected[dedup_key] = row
                added += 1
        return added

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
                add_rows(_parse_comment_items(payload, resolved_url))
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
        add_rows(_extract_dom_fallback(page, resolved_url))

        idle_scrolls = 0
        for step in range(1, max_scrolls + 1):
            if len(collected) >= max_comments:
                print(f"Đã đạt giới hạn {max_comments} bình luận.")
                break

            before_count = len(collected)
            for comment_sel in (
                'div[data-e2e="comment-level-1"]',
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
                add_rows(_extract_dom_fallback(page, resolved_url))

            after_count = len(collected)
            print(f"  Lần cuộn {step}/{max_scrolls}: thu được {after_count} bình luận")

            if after_count == before_count:
                idle_scrolls += 1
                if idle_scrolls >= 4:
                    print("Không tải thêm được bình luận mới sau 4 lần cuộn liên tiếp, dừng thu thập.")
                    break
            else:
                idle_scrolls = 0

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
    include_replies: bool = False,
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
    out_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "comment",
        "comment_id",
        "username",
        "nickname",
        "likes",
        "reply_count",
        "created_at",
        "post_url",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return rows, str(out_path)


def _normalize_comments_df(df: pd.DataFrame, comment_col: str) -> pd.DataFrame:
    """Chuẩn hóa DataFrame bình luận và giữ lại các cột metadata quan trọng."""
    out = df.copy()
    out["comment"] = out[comment_col].fillna("").astype(str).str.strip()
    out = out[out["comment"] != ""].reset_index(drop=True)

    defaults: dict[str, Any] = {
        "comment_id": "",
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
        elif col in {"likes", "reply_count"}:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
        else:
            out[col] = out[col].fillna("").astype(str)

    ordered = [
        "comment",
        "comment_id",
        "username",
        "nickname",
        "likes",
        "reply_count",
        "created_at",
        "post_url",
    ]
    extra_cols = [c for c in out.columns if c not in ordered and c != comment_col]
    return out[ordered + extra_cols]


def load_comments_dataframe(
    comments_source: str | Path | Sequence[str],
    text_column: str | None = None,
) -> pd.DataFrame:
    """Đọc bình luận từ list Python hoặc file CSV / JSON / TXT, giữ nguyên metadata."""
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
        df = pd.read_csv(path, encoding="utf-8-sig")
        if df.empty:
            return _normalize_comments_df(pd.DataFrame(columns=["comment"]), "comment")

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

    if ext == ".txt":
        with path.open("r", encoding="utf-8") as f:
            rows = [{"comment": line.strip()} for line in f if line.strip()]
        if not rows:
            return _normalize_comments_df(pd.DataFrame(columns=["comment"]), "comment")
        return _normalize_comments_df(pd.DataFrame(rows), "comment")

    raise ValueError(
        f"Định dạng file comment chưa hỗ trợ: {path.suffix} (chỉ hỗ trợ .csv, .json, .txt)"
    )
