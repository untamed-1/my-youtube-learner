#!/usr/bin/env python3
"""Build the video_analysis/ package from an analysis.json the agent wrote after watching the video.

The agent decides what matters (summary text, clip windows, categories); this module validates those
decisions against the real video, cuts the clips, takes the screenshots, and writes every file. Standard
library plus ffmpeg only.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from download import cut_clip, grab_screenshot
from frames import get_metadata, parse_time
from render import clean_summary, render_html, render_index, render_markdown, stamp
from runtime import configure_stdio

MANAGED = ('summaries', 'metadata', 'screenshots', 'video_snippets', 'index.html', 'index.json')
RELATIONS = ('primary', 'viewpoint', 'phase')
RESERVED = {'con', 'prn', 'aux', 'nul', *(f'com{i}' for i in range(1, 10)), *(f'lpt{i}' for i in range(1, 10))}

SCHEMA_DOC = """\
{
  "summary": {
    "title": "optional; defaults to the video title",
    "overview": ["One sentence per list item.", "..."],
    "sections": [
      {"heading": "Descriptive topic heading",
       "kind": "topic",
       "blocks": [
         {"kind": "takeaway",
          "sentences": ["Exactly one sentence per item; never join sentences in one string."],
          "bullets": ["Short bullet items for key information, observations, actions, takeaways."]}
       ]}
    ]
  },
  "clips": [
    {"title": "Descriptive name of the specific action",
     "category": "Main subject or activity, derived from THIS video",
     "subcategory": "More specific classification (optional)",
     "description": "Detailed explanation of what happens in the clip",
     "start": 83.4, "end": 91.0,
     "importance": 4,
     "relation": "primary",
     "screenshot_time": 87.0}
  ],
  "moments": [{"title": "Visually important moment without a clip", "time": 120.5}]
}
kind (summary sections and blocks): topic=blue main topics and general information; takeaway=green key
takeaways and findings; critical=red critical moments, warnings, essential actions; technical=orange technical
details, movements, procedures; insight=purple additional insights and supporting information.
start/end: seconds on the SOURCE video timeline (number, or "MM:SS.s" string) bounding only the action itself;
the packager adds context before and after. importance: 1-5. relation: primary (default), or viewpoint / phase
for a deliberate near-duplicate (another angle, a distinct phase of the same movement).
"""


def slug(text: str, limit: int = 60) -> str:
    """A filesystem-safe name that keeps letters from any script; never empty, never a Windows reserved name."""
    value = re.sub(r'[^\w]+', '_', str(text), flags=re.UNICODE).strip('_')[:limit].strip('_') or 'Untitled'
    return value + '_' if value.lower() in RESERVED else value


def _seconds(value, what: str):
    if isinstance(value, bool):
        raise ValueError(f'{what} must be a time')
    if isinstance(value, (int, float)):
        if value != value or value in (float('inf'), float('-inf')) or value < 0:
            raise ValueError(f'{what} must be finite and non-negative')
        return float(value)
    try:
        parsed = parse_time(str(value))
    except SystemExit:  # parse_time exits on bad input; one bad clip must not abort the package.
        raise ValueError(f'{what} {str(value)[:20]!r} is not a time (use seconds or MM:SS)') from None
    if parsed is None:
        raise ValueError(f'{what} is missing')
    return parsed


def load_analysis(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        raise SystemExit(f'Cannot read analysis file {path}: {type(exc).__name__}: {exc}') from None
    if not isinstance(data, dict) or not isinstance(data.get('summary'), dict):
        raise SystemExit('analysis.json must be an object with a "summary" object. Expected shape:\n' + SCHEMA_DOC)
    return data


def plan_clips(raw_clips, duration: float, *, pad: float = 1.0, max_clips: int = 12,
               min_len: float = 1.0, max_len: float = 120.0) -> tuple[list[dict], list[dict]]:
    """Validate the agent's clips against the real timeline. Returns (accepted in time order, rejected).

    Rejections are reported, never silent. Overlapping `primary` clips are redundant; the more important
    one wins. `viewpoint` and `phase` clips are exempt because the agent marked them as deliberately distinct.
    """
    candidates, rejected = [], []

    def reject(item, reason):
        rejected.append({'title': str(item.get('title') if isinstance(item, dict) else item)[:80], 'reason': reason})

    for item in raw_clips if isinstance(raw_clips, list) else []:
        if not isinstance(item, dict):
            reject(item, 'clip is not an object')
            continue
        missing = [k for k in ('title', 'category', 'description') if not str(item.get(k) or '').strip()]
        if missing:
            reject(item, f'missing {", ".join(missing)}')
            continue
        try:
            start, end = _seconds(item.get('start'), 'start'), _seconds(item.get('end'), 'end')
        except ValueError as exc:
            reject(item, str(exc))
            continue
        if start >= duration:
            reject(item, f'starts at {stamp(start)}, past the end of the video ({stamp(duration)})')
            continue
        end = min(end, duration)
        if end - start < min_len:
            reject(item, f'action shorter than {min_len:g}s ({max(end - start, 0):.2f}s)')
            continue
        if end - start > max_len:
            reject(item, f'action longer than {max_len:g}s; split it into phases')
            continue
        relation = item.get('relation') if item.get('relation') in RELATIONS else 'primary'
        try:
            importance = min(5, max(1, int(item.get('importance', 3))))
        except (TypeError, ValueError):
            importance = 3
        shot = None
        try:
            if item.get('screenshot_time') is not None:
                shot = _seconds(item['screenshot_time'], 'screenshot_time')
        except ValueError:
            shot = None
        if shot is None or not start <= shot <= end:
            shot = (start + end) / 2
        candidates.append({
            'title': str(item['title']).strip(), 'category': str(item['category']).strip(),
            'subcategory': str(item.get('subcategory') or '').strip() or 'General',
            'description': str(item['description']).strip(), 'action_start': start, 'action_end': end,
            'start': max(0.0, start - pad), 'end': min(duration, end + pad), 'importance': importance,
            'relation': relation, 'screenshot_time': shot})

    accepted = []
    for clip in sorted(candidates, key=lambda c: (-c['importance'], c['action_start'])):
        if len(accepted) >= max_clips:
            reject(clip['title'], f'over the {max_clips}-clip limit; lower importance')
            continue
        if clip['relation'] == 'primary':
            clash = next((a for a in accepted if a['relation'] == 'primary' and _overlap(a, clip) > 0.5), None)
            if clash:
                reject(clip['title'], f'redundant: overlaps "{clash["title"]}" by more than half')
                continue
        accepted.append(clip)
    return sorted(accepted, key=lambda c: c['action_start']), rejected


def _overlap(a: dict, b: dict) -> float:
    shared = min(a['action_end'], b['action_end']) - max(a['action_start'], b['action_start'])
    return max(0.0, shared) / min(a['action_end'] - a['action_start'], b['action_end'] - b['action_start'])


def _claim_root(root: Path, source_id: str) -> None:
    """Refuse to write into a directory that is not ours or belongs to a different video."""
    index = root / 'index.json'
    if index.is_file():
        try:
            previous = json.loads(index.read_text(encoding='utf-8')).get('source_id')
        except (OSError, ValueError):
            previous = None
        if previous not in (None, source_id):
            raise SystemExit(f'{root} already holds the analysis of a different video ({previous}). '
                             'Use another --analysis-dir for this one.')
    elif root.exists() and any(p.name not in ('.cache',) for p in root.iterdir()):
        raise SystemExit(f'{root} is not empty and is not a video_analysis package; choose an empty --analysis-dir.')
    for name in MANAGED:  # Clear only what this tool writes, so a rerun leaves no stale clips behind.
        target = root / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    root.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def build_package(analysis: dict, video_path: str, root: Path, *, metadata: dict, source: dict,
                  pad: float = 1.0, max_clips: int = 12, min_len: float = 1.0, max_len: float = 120.0,
                  cutter=cut_clip, shooter=grab_screenshot) -> dict:
    """Write the whole package. Returns a report; per-clip failures are recorded and do not stop the rest."""
    root = Path(root).resolve()
    probe = get_metadata(video_path)
    duration = probe['duration_seconds']
    title = (analysis['summary'].get('title') or metadata.get('title') or source.get('title') or 'Video analysis').strip()
    summary = clean_summary(analysis['summary'])
    clips, rejected = plan_clips(analysis.get('clips'), duration, pad=pad, max_clips=max_clips,
                                 min_len=min_len, max_len=max_len)
    source_id = str(metadata.get('id') or source.get('url') or source.get('title') or title)
    _claim_root(root, source_id)
    _write_json(root / 'index.json', {'source_id': source_id, 'status': 'building'})  # Marks the directory as ours.

    # Screenshots: one per clip (at its key moment) plus any extra moments, numbered in time order.
    shots = [{'title': c['title'], 'time': c['screenshot_time'], 'clip': c} for c in clips]
    for moment in analysis.get('moments') if isinstance(analysis.get('moments'), list) else []:
        try:
            when = _seconds(moment.get('time'), 'time')
            label = str(moment.get('title') or '').strip()
        except (ValueError, AttributeError):
            continue
        if label and when < duration:
            shots.append({'title': label, 'time': when, 'clip': None})
    failures = []
    kept_shots = []
    for shot in sorted(shots, key=lambda s: s['time']):
        rel = f'screenshots/important_moment_{len(kept_shots) + 1:02d}.png'
        try:
            shooter(video_path, shot['time'], root / rel)
        except SystemExit as exc:
            failures.append({'what': f'screenshot "{shot["title"]}"', 'error': str(exc)})
            continue
        shot['path'] = rel
        kept_shots.append(shot)
        if shot['clip'] is not None:
            shot['clip']['screenshot'] = rel

    # Folders: categories and subcategories come from the content, numbered by first appearance.
    categories: dict[str, dict] = {}
    done = []
    for number, clip in enumerate(clips, 1):
        cat = categories.setdefault(clip['category'], {'dir': f'{len(categories) + 1:02d}_{slug(clip["category"])}', 'subs': {}})
        sub = cat['subs'].setdefault(clip['subcategory'], f'{len(cat["subs"]) + 1:02d}_{slug(clip["subcategory"])}')
        folder = Path('video_snippets') / cat['dir'] / sub / f'Action_{number:02d}_{slug(clip["title"], 48)}'
        mp4 = folder / f'action_{number:02d}.mp4'
        info = folder / f'action_{number:02d}_info.json'
        try:
            cut = cutter(video_path, clip['start'], clip['end'], root / mp4)
        except SystemExit as exc:
            failures.append({'what': f'clip "{clip["title"]}"', 'error': str(exc)})
            continue
        clip.update(number=number, mp4=mp4.as_posix(), info=info.as_posix(), duration=cut['duration'],
                    has_audio=cut.get('has_audio'), size_bytes=cut.get('size_bytes'))
        _write_json(root / info, {
            'clip_title': clip['title'], 'category': clip['category'], 'subcategory': clip['subcategory'],
            'action_description': clip['description'],
            'start_timestamp': stamp(clip['start']), 'end_timestamp': stamp(clip['end']),
            'start_seconds': round(clip['start'], 3), 'end_seconds': round(clip['end'], 3),
            'duration_seconds': round(cut['duration'], 3),
            'action_start_seconds': round(clip['action_start'], 3), 'action_end_seconds': round(clip['action_end'], 3),
            'context_padding_seconds': pad, 'importance': clip['importance'], 'relation': clip['relation'],
            'has_audio': cut.get('has_audio'), 'file': mp4.name,
            'source_video': {'title': source.get('title') or metadata.get('title'),
                             'url': source.get('url') or metadata.get('webpage_url')},
            'related_screenshot': clip.get('screenshot')})
        done.append(clip)

    _write_json(root / 'metadata' / 'video_metadata.json', {
        **metadata, 'video_file': {k: probe.get(k) for k in ('duration_seconds', 'width', 'height', 'codec', 'has_audio', 'size_bytes')}})
    (root / 'summaries').mkdir(parents=True, exist_ok=True)
    (root / 'summaries' / 'video_summary.md').write_text(render_markdown(summary, done, metadata, title), encoding='utf-8')
    (root / 'summaries' / 'video_summary.html').write_text(render_html(summary, done, metadata, title), encoding='utf-8')
    files = {'Summary (HTML)': 'summaries/video_summary.html', 'Summary (Markdown)': 'summaries/video_summary.md',
             'Metadata (JSON)': 'metadata/video_metadata.json'}
    (root / 'index.html').write_text(render_index(done, kept_shots, metadata, title, files), encoding='utf-8')
    _write_json(root / 'index.json', {
        'source_id': source_id, 'title': title, 'source': source, 'files': files, 'duration_seconds': duration,
        'screenshots': [{'path': s['path'], 'title': s['title'], 'timestamp_seconds': round(s['time'], 3)} for s in kept_shots],
        'clips': [{'number': c['number'], 'title': c['title'], 'category': c['category'], 'subcategory': c['subcategory'],
                   'start_seconds': round(c['start'], 3), 'end_seconds': round(c['end'], 3), 'start_timestamp': stamp(c['start']),
                   'end_timestamp': stamp(c['end']), 'duration_seconds': round(c['duration'], 3), 'video': c['mp4'],
                   'info': c['info'], 'screenshot': c.get('screenshot')} for c in done],
        'rejected_clips': rejected, 'failures': failures})
    return {'root': str(root), 'clips': done, 'screenshots': kept_shots, 'rejected': rejected, 'failures': failures,
            'title': title, 'duration': duration}


def print_report(report: dict) -> None:
    print()
    print('# watch: analysis package')
    print()
    print(f'- **Package:** `{report["root"]}`')
    print(f'- **Master index:** `{Path(report["root"]) / "index.html"}`')
    print(f'- **Clips:** {len(report["clips"])} extracted, {len(report["rejected"])} rejected, '
          f'{len(report["failures"])} failed')
    print(f'- **Screenshots:** {len(report["screenshots"])}')
    for clip in report['clips']:
        print(f'  - Action {clip["number"]:02d} `{clip["category"]} / {clip["subcategory"]}`: {clip["title"]} '
              f'({stamp(clip["start"])} → {stamp(clip["end"])})')
    if report['rejected']:
        print()
        print('## Rejected clips (fix the analysis and rerun --package if any matter)')
        print()
        for item in report['rejected']:
            print(f'- {item["title"]}: {item["reason"]}')
    if report['failures']:
        print()
        print('## Failures')
        print()
        for item in report['failures']:
            print(f'- {item["what"]}: {item["error"]}')


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(prog='package', description='Build a video_analysis package from analysis.json.')
    ap.add_argument('analysis')
    ap.add_argument('--video', required=True, help='Local video file the timestamps refer to')
    ap.add_argument('--out', default='video_analysis')
    ap.add_argument('--clip-pad', type=float, default=1.0)
    ap.add_argument('--max-clips', type=int, default=12)
    args = ap.parse_args()
    video = Path(args.video).expanduser().resolve()
    report = build_package(load_analysis(Path(args.analysis)), str(video), Path(args.out).expanduser(),
                           metadata={'title': video.stem}, source={'title': video.name, 'url': str(video)},
                           pad=args.clip_pad, max_clips=args.max_clips)
    print_report(report)
    return 1 if report['failures'] else 0


if __name__ == '__main__':
    configure_stdio()
    raise SystemExit(main())
