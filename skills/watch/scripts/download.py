#!/usr/bin/env python3
"""Bounded captions and isolated per-run media downloads through yt-dlp."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from runtime import configure_stdio, diagnostic, run_text

VIDEO_EXTS = {'.mp4', '.mkv', '.webm', '.mov', '.m4v', '.avi', '.flv', '.wmv'}


def is_url(source: str) -> bool:
    parsed = urlparse(source)
    return not source.startswith('-') and parsed.scheme in ('http', 'https') and bool(parsed.netloc)


def resolve_local(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise SystemExit(f'File not found: {p}')
    return {'video_path': str(p), 'subtitle_path': None, 'info': {'title': p.name, 'url': str(p)}, 'downloaded': False}


def auth_args(cookies_file=None, cookies_from_browser=None) -> list[str]:
    if cookies_file and cookies_from_browser:
        raise SystemExit('Choose either --cookies or --cookies-from-browser, not both.')
    if cookies_from_browser:
        return ['--no-cookies', '--cookies-from-browser', cookies_from_browser]
    if cookies_file:
        return ['--no-cookies-from-browser', '--cookies', str(Path(cookies_file).expanduser().resolve())]
    return []


def _common(directory: Path, auth: list[str]) -> list[str]:
    # Escape literal percent signs in the directory, while retaining template fields.
    prefix = str(directory.resolve()).replace('%', '%%')
    return ['yt-dlp', '--no-playlist', '--no-simulate', *auth,
            '-o', f'{prefix}/video.%(ext)s',
            '-o', f'subtitle:{prefix}/video.%(ext)s',
            '-o', f'infojson:{prefix}/video.%(ext)s']


def network_diagnostic(stderr: str) -> str:
    detail = diagnostic(stderr)
    lower = detail.lower()
    if any(s in lower for s in ('javascript runtime', 'js runtime', 'ejs', 'challenge solver')):
        hint = 'Update yt-dlp with its owning package manager and check Deno/EJS using setup.py --json.'
    elif any(s in lower for s in ('sign in', 'login required', 'authentication required', 'confirm you’re not a bot', "confirm you're not a bot")):
        hint = 'This source requires authentication; supply an explicit cookie option if you have access.'
    elif '429' in lower or 'too many requests' in lower:
        hint = 'The service is rate limiting requests; wait before trying again.'
    elif re.search(r'\b403\b', lower) or 'forbidden' in lower:
        # Only the latest yt-dlp is supported; sites routinely break older releases with 403s.
        hint = 'No generic fix for a 403; update yt-dlp to its latest release and retry once.'
    elif any(s in lower for s in ('egress denied', 'not in the allowlist', 'blocked by network policy')):
        hint = 'This environment blocks the destination; check its network settings or use an accessible local file.'
    elif any(s in lower for s in ('certificate_verify_failed', 'certificate verify failed', 'unable to get local issuer certificate')):
        hint = 'Check your trusted CA configuration; keep TLS verification enabled.'
    else:
        hint = 'Check the original error and update the owning yt-dlp package if needed.'
    return f'{detail}\n{hint}'


def _run(cmd: list[str]):
    result = run_text(cmd)
    if result.returncode != 0:
        raise SystemExit(f'yt-dlp failed (exit {result.returncode}): {network_diagnostic(result.stderr)}')
    return result


def _new_run(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='run-', dir=out_dir)).resolve()


def _read_metadata(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(raw, dict) or raw.get('_type') in ('playlist', 'multi_video') or 'entries' in raw:
            raise ValueError('expected one video, not a playlist')
        return raw
    except (OSError, ValueError) as exc:
        raise SystemExit(f'yt-dlp metadata unavailable or invalid: {type(exc).__name__}') from None


def _summary(raw: dict, url: str) -> dict:
    return {'title': raw.get('title'), 'uploader': raw.get('uploader') or raw.get('channel'),
            'duration': raw.get('duration'), 'url': raw.get('webpage_url') or url}


_METADATA_FIELDS = (
    'id', 'title', 'description', 'upload_date', 'release_date', 'timestamp', 'duration', 'duration_string',
    'view_count', 'like_count', 'comment_count', 'channel', 'channel_id', 'channel_url',
    'channel_follower_count', 'uploader', 'uploader_id', 'uploader_url', 'categories', 'tags',
    'chapters', 'language', 'age_limit', 'live_status', 'availability', 'webpage_url', 'extractor_key',
    'thumbnail', 'width', 'height', 'fps', 'resolution',
)


def rich_metadata(raw: dict, url: str) -> dict:
    """The yt-dlp fields worth keeping in video_metadata.json; absent fields are omitted, not guessed."""
    out = {key: raw[key] for key in _METADATA_FIELDS if raw.get(key) not in (None, '', [])}
    out.setdefault('webpage_url', url)
    date = str(out.get('upload_date') or '')
    if re.fullmatch(r'\d{8}', date):
        out['upload_date_iso'] = f'{date[:4]}-{date[4:6]}-{date[6:]}'
    return out


def select_caption(info: dict, language: str = 'auto') -> dict | None:
    """Select at most one track and keep provenance separate from yt-dlp's clean JSON."""
    manual = {k: v for k, v in (info.get('subtitles') or {}).items() if v and k != 'live_chat'}
    automatic = {k: v for k, v in (info.get('automatic_captions') or {}).items() if v and k != 'live_chat'}
    youtube = str(info.get('extractor_key') or info.get('extractor') or '').lower().startswith('youtube')
    originals = sorted(k for k in automatic if youtube and k.endswith('-orig'))
    original = originals[0][:-5] if len(originals) == 1 else None
    target = language if language != 'auto' else original or info.get('language')
    def matches(key):
        base = key.removesuffix('-orig')
        if language != 'auto':
            return key == target or base == target or ('-' not in language and base.split('-')[0] == target)
        return target and (base == target or base.split('-')[0] == str(target).split('-')[0])
    def english_first(key):  # Unknown source language: prefer English over alphabetical order.
        return key.removesuffix('-orig').split('-')[0] != 'en', key
    for kind, tracks in (('manual', manual), ('automatic', automatic)):
        keys = sorted(tracks, key=english_first)
        if target:
            keys = sorted((k for k in keys if matches(k)), key=lambda k: (k != target, k != f'{target}-orig', k))
        if keys:
            key = keys[0]
            # Auto mode must prefer original ASR to translated/dubbed metadata.
            if kind == 'automatic' and language == 'auto' and originals:
                key = originals[0]
            base = key.removesuffix('-orig')
            provenance = 'original' if original and base.split('-')[0] == original.split('-')[0] else 'unknown'
            if language != 'auto' and original and base.split('-')[0] != original.split('-')[0]:
                provenance = 'requested translation'
            return {'key': key, 'language': base, 'kind': kind, 'provenance': provenance}
    # Unknown language evidence: select one available track, visibly labeled unknown.
    if language == 'auto':
        for kind, tracks in (('manual', manual), ('automatic', automatic)):
            if tracks:
                key = min(tracks, key=english_first)
                return {'key': key, 'language': key.removesuffix('-orig'), 'kind': kind, 'provenance': 'unknown'}
    return None


def fetch_captions(url: str, out_dir: Path, *, sub_lang='auto', cookies_file=None, cookies_from_browser=None) -> dict:
    auth = auth_args(cookies_file, cookies_from_browser)
    run = _new_run(out_dir)
    info_path = run / 'video.info.json'
    result = {'video_path': None, 'subtitle_path': None, 'info': {'url': url}, 'downloaded': False,
              'run_dir': str(run), 'info_path': None, 'caption_track': None, 'errors': [], 'metadata': {}}
    try:
        _run([*_common(run, auth), '--skip-download', '--write-info-json', '--no-write-subs', '--no-write-auto-subs', '--', url])
        info = _read_metadata(info_path)
        result.update(info=_summary(info, url), info_path=str(info_path), metadata=rich_metadata(info, url))
        track = select_caption(info, sub_lang)
        if not track:
            return result
        result['caption_track'] = track
        # --sub-langs accumulates across configs: explicitly clear inherited all.
        _run([*_common(run, auth), '--load-info-json', str(info_path), '--skip-download',
              '--no-write-info-json',
              '--write-subs' if track['kind'] == 'manual' else '--no-write-subs',
              '--write-auto-subs' if track['kind'] == 'automatic' else '--no-write-auto-subs',
              '--sub-langs', '-all,^' + re.escape(track['key']) + '$',
              '--sub-format', 'vtt/best', '--convert-subs', 'vtt'])
        subtitle = (run / f"video.{track['key']}.vtt").resolve()
        if not subtitle.is_relative_to(run) or not subtitle.is_file() or not subtitle.stat().st_size:
            raise SystemExit('yt-dlp did not produce the selected caption track.')
        result['subtitle_path'] = str(subtitle)
    except SystemExit as exc:
        result['errors'].append(str(exc))
        print(f'[watch] captions unavailable: {exc}', file=sys.stderr)
    return result


QUALITIES = {
    'default': 'bv*[height<=720]+ba/b[height<=720]/bv+ba/b',  # Frames and transcripts do not need more.
    'best': 'bv*+ba/b',  # Clips cut from the source cannot exceed the downloaded quality.
}


def download_url(url: str, out_dir: Path, audio_only: bool = False, *, context=None, quality: str = 'default',
                 cookies_file=None, cookies_from_browser=None) -> dict:
    auth = auth_args(cookies_file, cookies_from_browser)
    context = context or {}
    run = Path(context['run_dir']) if context.get('run_dir') else _new_run(out_dir)
    record = run / 'final-path.jsonl'
    record.unlink(missing_ok=True)
    if quality not in QUALITIES:
        raise SystemExit(f'--quality must be one of: {", ".join(QUALITIES)}')
    fmt = 'ba/bestaudio' if audio_only else QUALITIES[quality]
    cmd = [*_common(run, auth), '-N', '8', '-f', fmt, '--merge-output-format', 'mp4',
           '--no-ignore-errors', '--no-write-subs', '--no-write-auto-subs', '--write-info-json',
           '--print-to-file', 'after_move:%(filepath)j', str(record).replace('%', '%%')]
    if context.get('info_path'):
        cmd += ['--load-info-json', context['info_path']]
    else:
        cmd += ['--', url]
    _run(cmd)
    try:
        lines = record.read_text(encoding='utf-8').splitlines()
        if len(lines) != 1:
            raise ValueError()
        value = json.loads(lines[0])
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError()
        video = Path(value).resolve()
        if (not video.is_relative_to(run.resolve()) or not video.is_file() or video.stat().st_size == 0
                or video.suffix in ('.part', '.ytdl', '.json', '.vtt') or re.search(r'\.f\d+\.', video.name)):
            raise ValueError()
    except (OSError, ValueError, TypeError):
        raise SystemExit('yt-dlp did not report one completed, nonempty media file inside this run directory.') from None
    if context.get('info'):
        info, metadata = context['info'], context.get('metadata') or {}
    else:
        raw = _read_metadata(run / 'video.info.json')
        info, metadata = _summary(raw, url), rich_metadata(raw, url)
    return {**context, 'video_path': str(video), 'subtitle_path': context.get('subtitle_path'),
            'info': info, 'metadata': metadata, 'downloaded': True, 'run_dir': str(run)}


def _cache_entry(url: str, quality: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / hashlib.sha256(f'{quality}\0{url}'.encode()).hexdigest()[:16]


def cache_lookup(url: str, quality: str, cache_dir: Path) -> dict | None:
    """A cached download, only if its recorded file still exists and is nonempty."""
    entry = _cache_entry(url, quality, cache_dir)
    try:
        record = json.loads((entry / 'media.json').read_text(encoding='utf-8'))
        video = (entry / record['video_name']).resolve()
        if video.parent != entry.resolve() or not video.is_file() or video.stat().st_size == 0:
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return {'video_path': str(video), 'subtitle_path': None, 'info': record.get('info') or {'url': url},
            'metadata': record.get('metadata') or {}, 'downloaded': True, 'cached': True, 'run_dir': str(entry)}


def cache_store(url: str, quality: str, cache_dir: Path, result: dict) -> dict:
    """Move a completed download into the cache; the record is written last so a crash leaves no entry."""
    entry = _cache_entry(url, quality, cache_dir)
    shutil.rmtree(entry, ignore_errors=True)
    entry.mkdir(parents=True)
    source = Path(result['video_path'])
    name = f'video{source.suffix}'
    shutil.move(str(source), str(entry / name))
    record = {'video_name': name, 'source_url': url, 'quality': quality,
              'info': result.get('info'), 'metadata': result.get('metadata')}
    (entry / 'media.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    return {**result, 'video_path': str(entry / name), 'cached': False, 'cache_entry': str(entry)}


def download(source: str, out_dir: Path, audio_only: bool = False, *, cache_dir=None, quality: str = 'default',
             **kwargs) -> dict:
    """Local paths resolve in place. URLs download fresh per run unless a cache_dir is given.

    The cache is opt-in so the default path keeps its stale-file guarantees; only complete,
    validated video downloads are ever stored (never audio-only).
    """
    if not is_url(source):
        return resolve_local(source)
    use_cache = cache_dir is not None and not audio_only
    if use_cache:
        hit = cache_lookup(source, quality, Path(cache_dir))
        if hit:
            return {**kwargs.get('context', {}), **hit}
    result = download_url(source, out_dir, audio_only, quality=quality, **kwargs)
    return cache_store(source, quality, Path(cache_dir), result) if use_cache else result


def probe_times(video: Path) -> tuple[float, float]:
    """(video_seconds, audio_seconds) stream durations; 0 when a stream is absent."""
    result = run_text(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams', str(video)], timeout=30)
    try:
        streams = json.loads(result.stdout)['streams']
        pick = lambda kind: next((float(s.get('duration') or 0) for s in streams if s.get('codec_type') == kind), 0.0)
        return pick('video'), pick('audio')
    except (ValueError, KeyError, TypeError):
        raise SystemExit(f'ffprobe could not read {video.name}: {diagnostic(result.stderr)}') from None


def cut_clip(video: str | Path, start: float, end: float, dest: Path, *, crf: int = 16, preset: str = 'medium') -> dict:
    """Frame-accurate .mp4 of [start, end] seconds. Re-encodes (stream copy would snap to keyframes).

    Input-side -ss seeks fast and, because the stream is re-encoded, still lands on the exact frame.
    Video and audio are cut by the same -t, so they stay in sync; audio is kept when present.
    """
    if not (0 <= start < end):
        raise SystemExit(f'Clip needs 0 <= start < end; got {start:.3f}-{end:.3f}s')
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = run_text(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                       '-ss', f'{start:.3f}', '-i', str(Path(video).resolve()), '-t', f'{end - start:.3f}',
                       '-map', '0:v:0', '-map', '0:a:0?',
                       '-c:v', 'libx264', '-preset', preset, '-crf', str(crf), '-pix_fmt', 'yuv420p',
                       '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(dest)])
    if result.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise SystemExit(f'ffmpeg could not cut {dest.name}: {diagnostic(result.stderr)}')
    video_s, audio_s = probe_times(dest)
    return {'path': str(dest), 'start': start, 'end': end, 'duration': video_s or end - start,
            'has_audio': audio_s > 0, 'size_bytes': dest.stat().st_size}


def grab_screenshot(video: str | Path, at: float, dest: Path) -> dict:
    """One lossless full-resolution PNG at an exact source timestamp."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = run_text(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', f'{max(at, 0):.3f}',
                       '-i', str(Path(video).resolve()), '-frames:v', '1', str(dest)])
    if result.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise SystemExit(f'ffmpeg produced no frame at {at:.2f}s: {diagnostic(result.stderr)}')
    return {'path': str(dest), 'timestamp_seconds': at}


if __name__ == '__main__':
    configure_stdio()
    if len(sys.argv) != 3:
        raise SystemExit('usage: download.py <url-or-path> <out-dir>')
    print(json.dumps(download(sys.argv[1], Path(sys.argv[2])), indent=2))
