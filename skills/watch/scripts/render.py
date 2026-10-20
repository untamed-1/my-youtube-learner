#!/usr/bin/env python3
"""Render a video analysis as Markdown, HTML, and a master index.

Spacing and color are decided here, from the `kind` of each block, never by the model that wrote the
summary. Markdown cannot express line spacing or font color, so the .md file uses emoji markers and
blank lines; the .html file carries the exact CSS. Every string from the video or the analysis is escaped.
"""
from __future__ import annotations

import html
import re
from urllib.parse import quote

KINDS = ('topic', 'takeaway', 'critical', 'technical', 'insight')
MARKERS = {'topic': '🔵', 'takeaway': '🟢', 'critical': '🔴', 'technical': '🟠', 'insight': '🟣'}
LABELS = {'topic': 'Topic', 'takeaway': 'Key takeaway', 'critical': 'Critical', 'technical': 'Technical detail',
          'insight': 'Insight'}

CSS = """
:root{--line:2em;--sentence-gap:var(--line);--topic-gap:calc(var(--line)*3.5);
--bg:#ffffff;--fg:#1f2937;--muted:#4b5563;--rule:#d1d5db;
--topic:#1d4ed8;--takeaway:#15803d;--critical:#b91c1c;--technical:#c2410c;--insight:#7e22ce}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f172a;--fg:#e5e7eb;--muted:#9ca3af;--rule:#334155;
--topic:#60a5fa;--takeaway:#4ade80;--critical:#f87171;--technical:#fb923c;--insight:#c084fc}}
:root[data-theme="dark"]{--bg:#0f172a;--fg:#e5e7eb;--muted:#9ca3af;--rule:#334155;
--topic:#60a5fa;--takeaway:#4ade80;--critical:#f87171;--technical:#fb923c;--insight:#c084fc}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/2 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:48rem;margin:0 auto;padding:2rem 16px 4rem}
h1,h2,h3{line-height:1.4;margin:0 0 var(--sentence-gap)}
h1{font-size:1.9rem}h2{font-size:1.4rem}h3{font-size:1.1rem}
p{margin:0}
p.sentence{margin:0 0 var(--sentence-gap)}
ul{margin:0 0 var(--sentence-gap);padding-left:1.4rem}
li{margin:0}
section.topic{margin-block-end:var(--topic-gap)}
.block{margin-block-end:var(--topic-gap)}
section.topic>.block:last-child{margin-block-end:0}
.kind-topic{color:var(--topic)}.kind-takeaway{color:var(--takeaway)}.kind-critical{color:var(--critical)}
.kind-technical{color:var(--technical)}.kind-insight{color:var(--insight)}
.label{font-weight:700;margin:0 0 var(--sentence-gap)}
.meta{color:var(--muted)}
a{color:var(--topic)}
a:focus-visible,video:focus-visible{outline:3px solid var(--topic);outline-offset:2px}
img,video{max-width:100%;height:auto;border-radius:6px}
.clip{margin-block-end:var(--topic-gap)}
.clip .meta{margin:0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(14rem,1fr));gap:1rem}
.grid figure{margin:0}.grid figcaption{font-size:.875rem;line-height:1.5;color:var(--muted)}
"""


def clean_summary(raw) -> dict:
    """Coerce a model-written summary into {title, overview, sections}; text only, unknown kinds become 'topic'."""
    raw = raw if isinstance(raw, dict) else {}

    def kind(value, default='topic'):
        return value if value in KINDS else default

    def strings(value):
        return [str(v).strip() for v in value if isinstance(v, (str, int, float)) and str(v).strip()] \
            if isinstance(value, list) else []

    sections = []
    for section in raw.get('sections') if isinstance(raw.get('sections'), list) else []:
        if not isinstance(section, dict) or not str(section.get('heading') or '').strip():
            continue
        blocks = []
        for block in section.get('blocks') if isinstance(section.get('blocks'), list) else []:
            if not isinstance(block, dict):
                continue
            sentences, bullets = strings(block.get('sentences')), strings(block.get('bullets'))
            if sentences or bullets:
                blocks.append({'kind': kind(block.get('kind')), 'sentences': sentences, 'bullets': bullets})
        if blocks:
            sections.append({'heading': str(section['heading']).strip(), 'kind': kind(section.get('kind')),
                             'blocks': blocks})
    overview = strings(raw.get('overview'))
    if not sections and not overview:
        raise SystemExit('analysis.summary needs at least one section with blocks, or an overview.')
    return {'title': str(raw.get('title') or '').strip(), 'overview': overview, 'sections': sections}


def stamp(seconds: float) -> str:
    """H:MM:SS.s or MM:SS.s, to a tenth of a second (frame-level precision matters for clips)."""
    tenths = int(round(seconds * 10))
    hours, rest = divmod(tenths, 36000)
    minutes, rest = divmod(rest, 600)
    secs = rest / 10
    return f'{hours}:{minutes:02d}:{secs:04.1f}' if hours else f'{minutes:02d}:{secs:04.1f}'


def _facts(metadata: dict) -> list[tuple[str, str]]:
    rows = [('Channel', metadata.get('channel') or metadata.get('uploader')),
            ('Uploaded', metadata.get('upload_date_iso')),
            ('Duration', metadata.get('duration_string') or (stamp(metadata['duration']) if metadata.get('duration') else None)),
            ('Views', f"{metadata['view_count']:,}" if isinstance(metadata.get('view_count'), int) else None),
            ('Source', metadata.get('webpage_url'))]
    return [(k, str(v)) for k, v in rows if v]


def _grouped(clips: list[dict]) -> list[tuple[str, list[tuple[str, list[dict]]]]]:
    """Clips as [(category, [(subcategory, [clip...])...])...], in first-appearance order."""
    tree: dict[str, dict[str, list[dict]]] = {}
    for clip in clips:
        tree.setdefault(clip['category'], {}).setdefault(clip['subcategory'], []).append(clip)
    return [(cat, list(subs.items())) for cat, subs in tree.items()]


def render_markdown(summary: dict, clips: list[dict], metadata: dict, title: str) -> str:
    out = [f'# {MARKERS["topic"]} {title}', '']
    for key, value in _facts(metadata):
        out.append(f'- {MARKERS["insight"]} **{key}:** {value}')
    out += ['', '<br><br><br>', '']
    if summary['overview']:
        out += [f'## {MARKERS["topic"]} Overview', '']
        for sentence in summary['overview']:
            out += [f'{MARKERS["topic"]} {sentence}', '']
        out += ['<br><br><br>', '']
    for section in summary['sections']:
        out += [f'## {MARKERS[section["kind"]]} {section["heading"]}', '']
        for block in section['blocks']:
            marker = MARKERS[block['kind']]
            out += [f'{marker} **{LABELS[block["kind"]]}**', '']
            for sentence in block['sentences']:
                out += [f'{marker} {sentence}', '']  # One sentence per paragraph: blank line = double spacing.
            for bullet in block['bullets']:
                out.append(f'- {marker} {bullet}')
            if block['bullets']:
                out.append('')
            out += ['<br><br><br>', '']
    if clips:
        out += [f'## {MARKERS["technical"]} Extracted clips', '']
        for category, subs in _grouped(clips):
            out += [f'### {MARKERS["topic"]} {category}', '']
            for sub, items in subs:
                out += [f'#### {MARKERS["insight"]} {sub}', '']
                for c in items:
                    out.append(f'- {MARKERS["technical"]} **{c["title"]}** — {stamp(c["start"])} → {stamp(c["end"])} '
                               f'({c["duration"]:.1f}s) — `{c["mp4"]}`')
                out.append('')
    out += ['_Markdown cannot set line spacing or font color; open `video_summary.html` for the exact layout._', '']
    return '\n'.join(out)


def _e(text) -> str:
    return html.escape(str(text), quote=True)


def _href(rel: str) -> str:
    return quote(rel.replace('\\', '/'), safe='/')


def _safe_link(url: str) -> str:
    return _e(url) if re.match(r'https?://', url or '', re.I) else ''


def _page(title: str, body: str) -> str:
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{_e(title)}</title>\n<style>{CSS}</style>\n</head>\n<body>\n<main>\n{body}\n</main>\n</body>\n</html>\n')


def _sentences_html(sentences: list[str], kind: str) -> str:
    return ''.join(f'<p class="sentence kind-{kind}">{_e(s)}</p>\n' for s in sentences)


def render_html(summary: dict, clips: list[dict], metadata: dict, title: str, *, link_prefix: str = '../') -> str:
    """The exact-layout summary. `link_prefix` points from summaries/ back to the package root."""
    body = [f'<h1 class="kind-topic">{_e(title)}</h1>']
    facts = []
    for key, value in _facts(metadata):
        link = _safe_link(value) if key == 'Source' else ''
        shown = f'<a href="{link}">{_e(value)}</a>' if link else _e(value)
        facts.append(f'<li><span class="kind-insight"><strong>{_e(key)}:</strong></span> {shown}</li>')
    if facts:
        body.append('<ul class="meta">' + ''.join(facts) + '</ul>')
    if summary['overview']:
        body.append('<section class="topic">\n<h2 class="kind-topic">Overview</h2>\n'
                    + _sentences_html(summary['overview'], 'topic') + '</section>')
    for section in summary['sections']:
        parts = [f'<section class="topic">\n<h2 class="kind-{section["kind"]}">{_e(section["heading"])}</h2>']
        for block in section['blocks']:
            kind = block['kind']
            inner = f'<p class="label kind-{kind}">{_e(LABELS[kind])}</p>\n' + _sentences_html(block['sentences'], kind)
            if block['bullets']:
                inner += (f'<ul class="kind-{kind}">' + ''.join(f'<li>{_e(b)}</li>' for b in block['bullets']) + '</ul>')
            parts.append(f'<div class="block">\n{inner}</div>')
        parts.append('</section>')
        body.append('\n'.join(parts))
    if clips:
        parts = ['<section class="topic">\n<h2 class="kind-technical">Extracted clips</h2>']
        for category, subs in _grouped(clips):
            parts.append(f'<h3 class="kind-topic">{_e(category)}</h3>')
            for sub, items in subs:
                parts.append(f'<p class="label kind-insight">{_e(sub)}</p>')
                for c in items:
                    href = _href(link_prefix + c['mp4'])
                    parts.append(
                        f'<p class="sentence kind-technical"><a href="{href}">{_e(c["title"])}</a> — '
                        f'{stamp(c["start"])} → {stamp(c["end"])} ({c["duration"]:.1f}s)</p>')
        parts.append('</section>')
        body.append('\n'.join(parts))
    return _page(title, '\n'.join(body))


def render_index(clips: list[dict], shots: list[dict], metadata: dict, title: str, files: dict) -> str:
    """Master index at the package root: summaries, metadata, screenshots, and every clip with its timestamps."""
    body = [f'<h1 class="kind-topic">{_e(title)}</h1>']
    links = ''.join(f'<li><a href="{_href(path)}">{_e(label)}</a></li>' for label, path in files.items())
    body.append(f'<ul>{links}</ul>')
    facts = ''.join(f'<li><strong>{_e(k)}:</strong> {_e(v)}</li>' for k, v in _facts(metadata))
    if facts:
        body.append(f'<ul class="meta">{facts}</ul>')
    if shots:
        figures = ''.join(
            f'<figure><a href="{_href(s["path"])}"><img loading="lazy" src="{_href(s["path"])}" alt="{_e(s["title"])}"></a>'
            f'<figcaption>{_e(s["title"])} · {stamp(s["time"])}</figcaption></figure>' for s in shots)
        body.append(f'<section class="topic"><h2 class="kind-takeaway">Screenshots</h2><div class="grid">{figures}</div></section>')
    if clips:
        parts = ['<section class="topic"><h2 class="kind-technical">Clips</h2>']
        for category, subs in _grouped(clips):
            parts.append(f'<h3 class="kind-topic">{_e(category)}</h3>')
            for sub, items in subs:
                parts.append(f'<p class="label kind-insight">{_e(sub)}</p>')
                for c in items:
                    poster = f' poster="{_href(c["screenshot"])}"' if c.get('screenshot') else ''
                    parts.append(
                        f'<div class="clip"><p class="label kind-technical">{_e(c["title"])}</p>'
                        f'<p class="meta">{stamp(c["start"])} → {stamp(c["end"])} · {c["duration"]:.1f}s · '
                        f'<a href="{_href(c["info"])}">info.json</a></p>'
                        f'<p class="meta">{_e(c["description"])}</p>'
                        f'<video controls preload="none"{poster} src="{_href(c["mp4"])}"></video></div>')
        parts.append('</section>')
        body.append('\n'.join(parts))
    return _page(f'{title} — index', '\n'.join(body))
