#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class Job:
    mkv_path: pathlib.Path
    ass_files: List[pathlib.Path]


def _iter_mkvs(episodes_done: pathlib.Path) -> Iterable[pathlib.Path]:
    for p in sorted(episodes_done.glob("*.mkv")):
        if p.is_file():
            yield p


def _find_ass_files(ass_root: pathlib.Path, mkv_stem: str) -> List[pathlib.Path]:
    d = ass_root / mkv_stem
    if not d.is_dir():
        return []
    return sorted([p for p in d.glob("*.ass") if p.is_file()])


def _track_name_for_ass(mkv_stem: str, ass_path: pathlib.Path) -> str:
    stem = ass_path.stem
    prefix = mkv_stem + "."
    suffix = stem[len(prefix) :] if stem.startswith(prefix) else stem
    if suffix and suffix != mkv_stem:
        return f"German ({suffix})"
    return "German"


def _build_mkvmerge_cmd(
    mkv_path: pathlib.Path,
    ass_files: Sequence[pathlib.Path],
    out_path: pathlib.Path,
    language: str,
    set_default_first: bool,
) -> List[str]:
    cmd: List[str] = [
        "mkvmerge",
        "-o",
        str(out_path),
        "--no-subtitles",
        str(mkv_path),
    ]
    for i, ass_path in enumerate(ass_files):
        default_flag = "yes" if (set_default_first and i == 0) else "no"
        cmd += [
            "--sub-charset",
            "0:utf-8",
            "--language",
            f"0:{language}",
            "--track-name",
            f"0:{_track_name_for_ass(mkv_path.stem, ass_path)}",
            "--default-track",
            f"0:{default_flag}",
            str(ass_path),
        ]
    return cmd


def _run(cmd: Sequence[str], *, dry_run: bool) -> None:
    if dry_run:
        print("DRY-RUN:", " ".join([shlex_quote(x) for x in cmd]))
        return
    subprocess.run(list(cmd), check=True)


def shlex_quote(s: str) -> str:
    # Minimal shlex.quote equivalent to keep output readable on Windows/WSL.
    if not s:
        return "''"
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@%_-+=:,./"
    if all(c in safe for c in s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def remux_one(
    job: Job,
    *,
    dry_run: bool,
    language: str,
    set_default_first: bool,
    keep_backup: bool,
    backup_suffix: str,
) -> None:
    if not job.ass_files:
        print(f"SKIP (no .ass): {job.mkv_path.name}")
        return

    mkv_path = job.mkv_path
    mkv_dir = mkv_path.parent

    with tempfile.NamedTemporaryFile(
        prefix=mkv_path.stem + ".remux.",
        suffix=".mkv",
        dir=str(mkv_dir),
        delete=False,
    ) as tmp:
        tmp_path = pathlib.Path(tmp.name)

    try:
        cmd = _build_mkvmerge_cmd(
            mkv_path=mkv_path,
            ass_files=job.ass_files,
            out_path=tmp_path,
            language=language,
            set_default_first=set_default_first,
        )
        _run(cmd, dry_run=dry_run)

        if dry_run:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return

        if keep_backup:
            backup_path = mkv_path.with_name(mkv_path.name + backup_suffix)
            if backup_path.exists():
                raise FileExistsError(f"Backup already exists: {backup_path}")
            os.replace(mkv_path, backup_path)

        os.replace(tmp_path, mkv_path)
        print(f"OK: {mkv_path.name} (+{len(job.ass_files)} ASS)")
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Remux OCR'd .ass subtitles back into episodes_done/*.mkv using mkvmerge (no re-encode)."
    )
    ap.add_argument("--episodes-done", type=pathlib.Path, default=pathlib.Path("episodes_done"))
    ap.add_argument("--ass-root", type=pathlib.Path, default=pathlib.Path("episodes_done") / "ass")
    ap.add_argument("--language", default="deu", help="Subtitle language code for mkvmerge (default: deu).")
    ap.add_argument(
        "--set-default-first",
        action="store_true",
        help="Mark the first added ASS track as default.",
    )
    ap.add_argument(
        "--backup",
        action="store_true",
        help="Keep a backup of the original MKV (adds .bak.mkv by default).",
    )
    ap.add_argument(
        "--backup-suffix",
        default=".bak.mkv",
        help="Backup filename suffix when using --backup (default: .bak.mkv).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print mkvmerge commands only; do not write outputs.")
    args = ap.parse_args()

    if shutil.which("mkvmerge") is None:
        print("ERROR: mkvmerge not found. Install mkvtoolnix (mkvmerge) and try again.", file=sys.stderr)
        return 2

    episodes_done = args.episodes_done.resolve()
    ass_root = args.ass_root.resolve()

    mkvs = list(_iter_mkvs(episodes_done))
    if not mkvs:
        print(f"ERROR: No .mkv files found in {episodes_done}", file=sys.stderr)
        return 2

    jobs: List[Job] = []
    for mkv_path in mkvs:
        ass_files = _find_ass_files(ass_root, mkv_path.stem)
        jobs.append(Job(mkv_path=mkv_path, ass_files=ass_files))

    for job in jobs:
        remux_one(
            job,
            dry_run=args.dry_run,
            language=args.language,
            set_default_first=args.set_default_first,
            keep_backup=args.backup,
            backup_suffix=args.backup_suffix,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
