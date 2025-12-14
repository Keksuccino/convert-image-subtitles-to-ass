#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image


PROMPT = "Please return the text in the image. Only the text, nothing else."


@dataclasses.dataclass(frozen=True)
class TrackInfo:
    id: int
    language: Optional[str]
    default: Optional[bool]
    forced: Optional[bool]
    name: Optional[str]
    codec: str


@dataclasses.dataclass
class SubtitleEvent:
    start_s: float
    end_s: float
    image: Image.Image
    text: Optional[str] = None


def load_dotenv(dotenv_path: pathlib.Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def run(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.stdout


def list_mkv_pgs_tracks(mkv_path: pathlib.Path) -> List[TrackInfo]:
    raw = run(["mkvmerge", "-J", str(mkv_path)])
    data = json.loads(raw)
    tracks: List[TrackInfo] = []
    for t in data.get("tracks", []):
        if t.get("type") != "subtitles":
            continue
        codec = str(t.get("codec") or "")
        if "PGS" not in codec and "hdmv_pgs" not in codec.lower():
            continue
        props = t.get("properties") or {}
        tracks.append(
            TrackInfo(
                id=int(t["id"]),
                language=props.get("language"),
                default=props.get("default_track"),
                forced=props.get("forced_track"),
                name=props.get("track_name"),
                codec=codec,
            )
        )
    return tracks


def filter_tracks(tracks: List[TrackInfo], policy: str, include_ids: Optional[List[int]]) -> List[TrackInfo]:
    if include_ids:
        wanted = set(include_ids)
        return [t for t in tracks if t.id in wanted]

    policy = policy.lower()
    if policy == "all":
        return tracks
    if policy == "default":
        return [t for t in tracks if t.default]
    if policy == "forced":
        return [t for t in tracks if t.forced]
    if policy in ("default_or_forced", "default-or-forced"):
        return [t for t in tracks if t.default or t.forced]
    raise ValueError(f"Unknown track policy: {policy}")


def mkvextract_track_to_sup(mkv_path: pathlib.Path, track_id: int, out_sup: pathlib.Path, show_progress: bool = True) -> None:
    out_sup.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["mkvextract", "tracks", str(mkv_path), f"{track_id}:{out_sup}"],
        check=True,
        stdout=None if show_progress else subprocess.DEVNULL,
        stderr=None if show_progress else subprocess.DEVNULL,
    )


def _u8(b: bytes, i: int) -> int:
    return b[i]


def _u16be(b: bytes, i: int) -> int:
    return (b[i] << 8) | b[i + 1]


def _u24be(b: bytes, i: int) -> int:
    return (b[i] << 16) | (b[i + 1] << 8) | b[i + 2]


def _u32be(b: bytes, i: int) -> int:
    return (b[i] << 24) | (b[i + 1] << 16) | (b[i + 2] << 8) | b[i + 3]


def _clamp255(x: float) -> int:
    if x < 0:
        return 0
    if x > 255:
        return 255
    return int(round(x))


def ycbcr_to_rgba(y: int, cr: int, cb: int, a: int) -> Tuple[int, int, int, int]:
    # ITU-R BT.601-ish conversion used by most implementations for PGS palette.
    yf = float(y)
    crf = float(cr) - 128.0
    cbf = float(cb) - 128.0
    r = _clamp255(yf + 1.4020 * crf)
    g = _clamp255(yf - 0.344136 * cbf - 0.714136 * crf)
    b = _clamp255(yf + 1.7720 * cbf)
    return (r, g, b, int(a) & 0xFF)


def decode_pgs_rle(width: int, height: int, rle: bytes) -> bytes:
    """
    Decode PGS bitmap RLE into palette indices (len == width*height).
    Based on FFmpeg's pgssubdec.c decode_rle().
    """
    out = bytearray(width * height)
    pixel_count = 0
    line_count = 0
    i = 0

    while i < len(rle) and line_count < height:
        color = rle[i]
        i += 1
        run_len = 1

        if color == 0x00:
            if i >= len(rle):
                break
            flags = rle[i]
            i += 1
            run_len = flags & 0x3F
            if flags & 0x40:
                if i >= len(rle):
                    break
                run_len = (run_len << 8) + rle[i]
                i += 1
            color = rle[i] if (flags & 0x80) else 0
            if flags & 0x80:
                if i >= len(rle):
                    break
                i += 1

        if run_len > 0:
            end = pixel_count + run_len
            if end > len(out):
                break
            out[pixel_count:end] = bytes([color]) * run_len
            pixel_count = end
        else:
            # New line marker. Align to next row if needed.
            if width > 0:
                rem = pixel_count % width
                if rem:
                    pad = min(width - rem, len(out) - pixel_count)
                    if pad > 0:
                        out[pixel_count : pixel_count + pad] = b"\x00" * pad
                        pixel_count += pad
            line_count += 1

    return bytes(out)


@dataclasses.dataclass
class _PCSObjectRef:
    object_id: int
    x: int
    y: int
    crop: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)


@dataclasses.dataclass
class _PCS:
    pts_s: float
    width: int
    height: int
    palette_id: int
    objects: List[_PCSObjectRef]


def parse_sup_to_events(sup_path: pathlib.Path, tail_duration_s: float = 2.0) -> List[SubtitleEvent]:
    blob = sup_path.read_bytes()

    palettes: Dict[int, Dict[int, Tuple[int, int, int, int]]] = {}
    object_buffers: Dict[int, Dict[str, Any]] = {}
    objects: Dict[int, Dict[str, Any]] = {}

    current_pcs: Optional[_PCS] = None
    active_event: Optional[SubtitleEvent] = None
    events: List[SubtitleEvent] = []

    def render_pcs(pcs: _PCS) -> Image.Image:
        palette = palettes.get(pcs.palette_id) or {}

        obj_imgs: List[Tuple[Image.Image, int, int]] = []
        min_x: Optional[int] = None
        min_y: Optional[int] = None
        max_x: Optional[int] = None
        max_y: Optional[int] = None

        for ref in pcs.objects:
            obj = objects.get(ref.object_id)
            if not obj:
                continue
            w = int(obj["width"])
            h = int(obj["height"])
            idx = decode_pgs_rle(w, h, obj["rle"])

            rgba = bytearray(w * h * 4)
            for p in range(w * h):
                entry = palette.get(idx[p], (0, 0, 0, 0))
                rgba[p * 4 + 0] = entry[0]
                rgba[p * 4 + 1] = entry[1]
                rgba[p * 4 + 2] = entry[2]
                rgba[p * 4 + 3] = entry[3]

            im = Image.frombytes("RGBA", (w, h), bytes(rgba))
            if ref.crop:
                cx, cy, cw, ch = ref.crop
                im = im.crop((cx, cy, cx + cw, cy + ch))
                w, h = im.size

            obj_imgs.append((im, ref.x, ref.y))
            min_x = ref.x if min_x is None else min(min_x, ref.x)
            min_y = ref.y if min_y is None else min(min_y, ref.y)
            max_x = (ref.x + w) if max_x is None else max(max_x, ref.x + w)
            max_y = (ref.y + h) if max_y is None else max(max_y, ref.y + h)

        if min_x is None or min_y is None or max_x is None or max_y is None:
            return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

        out = Image.new("RGBA", (max_x - min_x, max_y - min_y), (0, 0, 0, 0))
        for im, x, y in obj_imgs:
            out.alpha_composite(im, dest=(x - min_x, y - min_y))

        bbox = out.getbbox()
        if bbox:
            out = out.crop(bbox)
        return out

    i = 0
    while i + 13 <= len(blob):
        if blob[i : i + 2] != b"PG":
            i += 1
            continue

        pts90 = _u32be(blob, i + 2)
        # dts90 = _u32be(blob, i + 6)
        seg_type = _u8(blob, i + 10)
        seg_len = _u16be(blob, i + 11)
        seg = blob[i + 13 : i + 13 + seg_len]
        i = i + 13 + seg_len

        pts_s = pts90 / 90000.0

        # PDS
        if seg_type == 0x14 and len(seg) >= 2:
            palette_id = _u8(seg, 0)
            # palette_version = _u8(seg, 1)
            pal = palettes.setdefault(palette_id, {})
            j = 2
            while j + 5 <= len(seg):
                entry_id = _u8(seg, j + 0)
                y = _u8(seg, j + 1)
                cr = _u8(seg, j + 2)
                cb = _u8(seg, j + 3)
                a = _u8(seg, j + 4)
                pal[entry_id] = ycbcr_to_rgba(y, cr, cb, a)
                j += 5
            continue

        # ODS
        if seg_type == 0x15 and len(seg) >= 11:
            object_id = _u16be(seg, 0)
            # object_version = _u8(seg, 2)
            seq_flags = _u8(seg, 3)
            first = (seq_flags & 0x80) != 0
            last = (seq_flags & 0x40) != 0
            # obj_data_len = _u24be(seg, 4)
            j = 7

            if first:
                if j + 4 > len(seg):
                    continue
                width = _u16be(seg, j)
                height = _u16be(seg, j + 2)
                j += 4
                object_buffers[object_id] = {"width": width, "height": height, "rle": bytearray()}

            buf = object_buffers.get(object_id)
            if not buf:
                continue

            buf["rle"].extend(seg[j:])
            if last:
                objects[object_id] = {"width": buf["width"], "height": buf["height"], "rle": bytes(buf["rle"])}
                object_buffers.pop(object_id, None)
            continue

        # PCS
        if seg_type == 0x16 and len(seg) >= 11:
            width = _u16be(seg, 0)
            height = _u16be(seg, 2)
            # frame_rate = _u8(seg, 4)
            # composition_number = _u16be(seg, 5)
            # composition_state = _u8(seg, 7)
            # palette_update_flag = _u8(seg, 8)
            palette_id = _u8(seg, 9)
            obj_count = _u8(seg, 10)

            objs: List[_PCSObjectRef] = []
            j = 11
            for _ in range(obj_count):
                if j + 8 > len(seg):
                    break
                object_id = _u16be(seg, j)
                # window_id = _u8(seg, j + 2)
                cropped_flag = _u8(seg, j + 3)
                x = _u16be(seg, j + 4)
                y = _u16be(seg, j + 6)
                j += 8
                crop: Optional[Tuple[int, int, int, int]] = None
                if cropped_flag == 0x40:
                    if j + 8 > len(seg):
                        break
                    cx = _u16be(seg, j)
                    cy = _u16be(seg, j + 2)
                    cw = _u16be(seg, j + 4)
                    ch = _u16be(seg, j + 6)
                    j += 8
                    crop = (cx, cy, cw, ch)
                objs.append(_PCSObjectRef(object_id=object_id, x=x, y=y, crop=crop))

            current_pcs = _PCS(pts_s=pts_s, width=width, height=height, palette_id=palette_id, objects=objs)
            continue

        # END: finalize current display set based on last PCS
        if seg_type == 0x80:
            if current_pcs is None:
                continue

            if len(current_pcs.objects) == 0:
                if active_event is not None:
                    active_event.end_s = current_pcs.pts_s
                    if active_event.end_s > active_event.start_s:
                        events.append(active_event)
                    active_event = None
            else:
                img = render_pcs(current_pcs)
                if active_event is not None:
                    active_event.end_s = current_pcs.pts_s
                    if active_event.end_s > active_event.start_s:
                        events.append(active_event)
                active_event = SubtitleEvent(start_s=current_pcs.pts_s, end_s=current_pcs.pts_s + tail_duration_s, image=img)

            current_pcs = None
            continue

    if active_event is not None:
        if active_event.end_s <= active_event.start_s:
            active_event.end_s = active_event.start_s + tail_duration_s
        events.append(active_event)

    events.sort(key=lambda e: (e.start_s, e.end_s))
    return events


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        timeout_s: float = 120.0,
        max_retries: int = 6,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.site_url = site_url
        self.app_name = app_name
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def ocr_one_image(self, png_bytes: bytes, prompt: str = PROMPT) -> str:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 256,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        }

        body = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read()
                data = json.loads(raw.decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError(f"Unexpected OpenRouter response content type: {type(content)}")
                return content.strip().replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\N").strip()
            except urllib.error.HTTPError as e:
                last_err = e
                retry_after = e.headers.get("Retry-After")
                wait_s = float(retry_after) if retry_after else min(2**attempt, 30.0)
                if e.code in (408, 429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(wait_s)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as e:
                last_err = e
                wait_s = min(2**attempt, 30.0)
                if attempt < self.max_retries:
                    time.sleep(wait_s)
                    continue
                raise RuntimeError("OpenRouter request failed after retries") from last_err

        raise RuntimeError("OpenRouter request failed") from last_err


def ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_cs = int(round(seconds * 100.0))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def write_ass(events: List[SubtitleEvent], out_ass: pathlib.Path, play_res: Tuple[int, int] = (1920, 1080)) -> None:
    out_ass.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("[Script Info]")
    lines.append("ScriptType: v4.00+")
    lines.append(f"PlayResX: {play_res[0]}")
    lines.append(f"PlayResY: {play_res[1]}")
    lines.append("ScaledBorderAndShadow: yes")
    lines.append("")
    lines.append("[V4+ Styles]")
    lines.append(
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
    )
    lines.append(
        "Style: Default,Arial Rounded MT Bold,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "0,0,0,0,100,100,0,0,1,3,0,2,60,60,40,1"
    )
    lines.append("")
    lines.append("[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    for ev in events:
        txt = ev.text or ""
        start = ass_time(ev.start_s)
        end = ass_time(ev.end_s)
        if end <= start:
            end = ass_time(ev.start_s + 0.50)
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{txt}")

    out_ass.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cache_key(png: bytes, model: str, prompt: str) -> str:
    h = hashlib.sha256()
    h.update(b"png-sha256:")
    h.update(hashlib.sha256(png).digest())
    h.update(b"\nmodel:")
    h.update(model.encode("utf-8"))
    h.update(b"\nprompt-sha256:")
    h.update(hashlib.sha256(prompt.encode("utf-8")).digest())
    return h.hexdigest()


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: pathlib.Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def iter_mkv_files(episodes_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    for p in sorted(episodes_dir.glob("*.mkv")):
        if p.is_file():
            yield p


def _fmt_s(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - (minutes * 60)
    if minutes < 60:
        return f"{minutes:d}m{rem:04.1f}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours:d}h{minutes:02d}m"


def _print_progress(prefix: str, done: int, total: int, start_t: float, extra: str = "") -> None:
    total = max(total, 1)
    done = max(0, min(done, total))
    pct = (done / total) * 100.0
    elapsed = max(0.001, time.time() - start_t)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else 0.0
    msg = f"{prefix} {done}/{total} ({pct:5.1f}%) {rate:5.2f}/s ETA {_fmt_s(eta)}"
    if extra:
        msg += f" {extra}"
    print("\r" + msg + " " * 10, end="", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert MKV PGS subtitles to ASS via OpenRouter vision OCR (1 image per request).")
    ap.add_argument("--episodes-dir", default="episodes", help="Directory containing .mkv files (default: episodes)")
    ap.add_argument("--out-dir", default="episodes/ass", help="Where to write .ass files (default: episodes/ass)")
    ap.add_argument("--work-dir", default=".work", help="Working directory for extracted .sup and caches (default: .work)")
    ap.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash"))
    ap.add_argument("--openrouter-base-url", default=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    ap.add_argument("--site-url", default=os.environ.get("OPENROUTER_SITE_URL"))
    ap.add_argument("--app-name", default=os.environ.get("OPENROUTER_APP_NAME", "subtitle_convert"))
    ap.add_argument("--tail-duration", type=float, default=2.0, help="Fallback duration when an end time is missing (seconds)")
    ap.add_argument(
        "--track-policy",
        default="all",
        choices=["default_or_forced", "default", "forced", "all"],
        help="Which PGS subtitle tracks to process (default: all)",
    )
    ap.add_argument(
        "--track-id",
        action="append",
        type=int,
        default=[],
        help="Process only this mkvmerge track id (repeatable). Overrides --track-policy.",
    )
    ap.add_argument("--no-progress", action="store_true", help="Disable progress output")
    ap.add_argument("--dry-run", action="store_true", help="Do not call OpenRouter; write ASS with empty text")
    ap.add_argument("--limit-images", type=int, default=0, help="Process only first N subtitle images per track (0 = no limit)")
    args = ap.parse_args()

    episodes_dir = pathlib.Path(args.episodes_dir)
    out_dir = pathlib.Path(args.out_dir)
    work_dir = pathlib.Path(args.work_dir)

    if not episodes_dir.exists():
        raise SystemExit(f"Episodes dir not found: {episodes_dir}")

    load_dotenv(pathlib.Path(".env"))
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    client: Optional[OpenRouterClient] = None
    if not args.dry_run:
        if not api_key:
            raise SystemExit("Missing OPENROUTER_API_KEY env var (or run with --dry-run).")
        client = OpenRouterClient(
            api_key=api_key,
            model=args.model,
            base_url=args.openrouter_base_url,
            site_url=args.site_url,
            app_name=args.app_name,
        )

    global_cache_path = work_dir / "openrouter_cache.json"
    cache: Dict[str, str] = {} if args.dry_run else load_json(global_cache_path)  # cache_key(png,model,prompt)->text

    mkvs = list(iter_mkv_files(episodes_dir))
    if not mkvs:
        print(f"No .mkv files found in {episodes_dir}", file=sys.stderr)
        return 2

    tasks: List[Tuple[pathlib.Path, TrackInfo]] = []
    for mkv in mkvs:
        tracks = filter_tracks(list_mkv_pgs_tracks(mkv), args.track_policy, args.track_id or None)
        for t in tracks:
            tasks.append((mkv, t))
    if not tasks:
        print("[skip] no matching PGS subtitle tracks found", file=sys.stderr)
        return 0

    for task_idx, (mkv, t) in enumerate(tasks, start=1):
        if not args.no_progress:
            print(f"[task {task_idx}/{len(tasks)}] {mkv.name} track{t.id}", file=sys.stderr)

            track_tag = f"track{t.id}"
            lang = t.language or "und"
            flags = []
            if t.default:
                flags.append("default")
            if t.forced:
                flags.append("forced")
            suffix = ".".join([lang] + flags) if flags else lang

            sup_path = work_dir / "sup" / mkv.stem / f"{track_tag}.sup"
            ass_path = out_dir / mkv.stem / f"{mkv.stem}.{track_tag}.{suffix}.ass"

            if not sup_path.exists():
                if not args.no_progress:
                    print(f"[extract] {mkv.name} {track_tag} -> {sup_path}", file=sys.stderr)
                mkvextract_track_to_sup(mkv, t.id, sup_path, show_progress=not args.no_progress)
            else:
                if not args.no_progress:
                    print(f"[extract] (cached) {sup_path}", file=sys.stderr)

            t0 = time.time()
            print(f"[parse] {sup_path}", file=sys.stderr)
            events = parse_sup_to_events(sup_path, tail_duration_s=float(args.tail_duration))
            if not args.no_progress:
                print(f"[parse] {len(events)} images in {_fmt_s(time.time() - t0)}", file=sys.stderr)
            if args.limit_images and args.limit_images > 0:
                events = events[: args.limit_images]

            print(f"[ocr] {mkv.name} {track_tag}: {len(events)} images", file=sys.stderr)
            ocr_start = time.time()
            cache_hits = 0
            for idx, ev in enumerate(events, start=1):
                png = pil_to_png_bytes(ev.image)
                key = cache_key(png, args.model, PROMPT)
                if key in cache:
                    ev.text = cache[key]
                    cache_hits += 1
                    if not args.no_progress:
                        _print_progress("[ocr]", idx, len(events), ocr_start, extra=f"(cache {cache_hits})")
                    continue

                if args.dry_run:
                    ev.text = ""
                else:
                    assert client is not None
                    ev.text = client.ocr_one_image(png)

                if not args.dry_run:
                    cache[key] = ev.text
                    if idx % 5 == 0:
                        save_json(global_cache_path, cache)

                if not args.no_progress:
                    _print_progress("[ocr]", idx, len(events), ocr_start, extra=f"(cache {cache_hits})")

            if not args.no_progress:
                print(file=sys.stderr)
            if not args.dry_run:
                save_json(global_cache_path, cache)
            write_ass(events, ass_path)
            print(f"[write] {ass_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
