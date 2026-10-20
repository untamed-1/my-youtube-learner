"""Analysis package: clip planning, ffmpeg cutting, rendering, folder layout, cache, and CLI wiring."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import download
import gemini
import package
import render

SCRIPT = Path(download.__file__).with_name('watch.py')
COLORS = ['red', 'green', 'blue', 'white', 'black', 'yellow', 'cyan', 'magenta', 'gray', 'orange', 'purple', 'brown', 'navy', 'olive']


def _ffmpeg(*args):
    result = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture(scope='session')
def av_clip(tmp_path_factory) -> Path:
    """5.6s, 14 solid colors 0.4s each, ONE keyframe (so keyframe-snapped cuts are detectable), plus a sine track."""
    path = tmp_path_factory.mktemp('av') / 'colors.mp4'
    inputs = [a for c in COLORS for a in ('-f', 'lavfi', '-t', '0.4', '-i', f'color=c={c}:s=320x240:r=10')]
    streams = ''.join(f'[{i}:v]' for i in range(len(COLORS)))
    _ffmpeg(*inputs, '-f', 'lavfi', '-i', 'sine=frequency=440:duration=5.6',
            '-filter_complex', f'{streams}concat=n={len(COLORS)}:v=1:a=0[v]', '-map', '[v]', '-map', f'{len(COLORS)}:a',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-g', '600', '-c:a', 'aac', str(path))
    return path


def _first_pixel(video: Path):
    out = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-frames:v', '1', '-vf', 'scale=1:1',
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout
    return tuple(out[:3])


# ---------- slug / planning ----------

@pytest.mark.parametrize('text,expected', [
    ('Squat: descent (slow)', 'Squat_descent_slow'), ('', 'Untitled'), ('???', 'Untitled'),
    ('CON', 'CON_'), ('スクワット 準備', 'スクワット_準備'), ('../../etc/passwd', 'etc_passwd'),
])
def test_slug(text, expected):
    assert package.slug(text) == expected


def test_slug_is_length_bounded():
    assert len(package.slug('a' * 500)) == 60


def _clip(title='Squat', start=10, end=14, **extra):
    return {'title': title, 'category': 'Lifting', 'description': 'Lifter squats.', 'start': start, 'end': end, **extra}


def test_plan_pads_clamps_and_sorts():
    accepted, rejected = package.plan_clips([_clip('B', 50, 200), _clip('A', '0:01', '0:03.5')], 60.0, pad=1.0)
    assert not rejected
    a, b = accepted
    assert (a['title'], a['start'], a['end'], a['action_start'], a['action_end']) == ('A', 0.0, 4.5, 1.0, 3.5)
    assert (b['start'], b['end']) == (49.0, 60.0)  # end clamped to the real duration, padding clamped to bounds


@pytest.mark.parametrize('clip,reason', [
    (_clip(start=99, end=120), 'past the end'), (_clip(start=5, end=5.4), 'shorter'),
    ({'title': 'x', 'category': 'c', 'start': 1, 'end': 3}, 'missing description'),
    (_clip(start='nope'), 'not a time'), (_clip(start=-3, end=4), 'non-negative'), ('not a dict', 'not an object'),
])
def test_plan_rejects_with_reason(clip, reason):
    accepted, rejected = package.plan_clips([clip], 60.0)
    assert accepted == [] and len(rejected) == 1 and reason in rejected[0]['reason']


def test_overlong_actions_are_rejected_not_truncated():
    accepted, rejected = package.plan_clips([_clip(start=0, end=300)], 500.0)
    assert accepted == [] and 'split it into phases' in rejected[0]['reason']


def test_overlapping_primary_clips_keep_the_more_important_one():
    accepted, rejected = package.plan_clips(
        [_clip('Minor', 10, 14, importance=2), _clip('Major', 11, 15, importance=5)], 60.0)
    assert [c['title'] for c in accepted] == ['Major']
    assert 'redundant' in rejected[0]['reason'] and 'Major' in rejected[0]['reason']


def test_viewpoint_and_phase_clips_are_exempt_from_redundancy():
    accepted, rejected = package.plan_clips(
        [_clip('Front', 10, 14), _clip('Side', 10, 14, relation='viewpoint'), _clip('Bottom', 12, 15, relation='phase')], 60.0)
    assert len(accepted) == 3 and not rejected


def test_max_clips_drops_the_least_important_and_says_so():
    clips = [_clip(f'C{i}', i * 10, i * 10 + 4, importance=i % 5 + 1) for i in range(6)]
    accepted, rejected = package.plan_clips(clips, 100.0, max_clips=4)
    assert len(accepted) == 4 and len(rejected) == 2
    assert all('limit' in r['reason'] for r in rejected)
    assert min(c['importance'] for c in accepted) >= 2


def test_screenshot_time_defaults_to_midpoint_and_stays_inside_the_action():
    (a,), _ = package.plan_clips([_clip(start=10, end=14)], 60.0)
    assert a['screenshot_time'] == 12.0
    (a,), _ = package.plan_clips([_clip(start=10, end=14, screenshot_time=13)], 60.0)
    assert a['screenshot_time'] == 13.0
    (a,), _ = package.plan_clips([_clip(start=10, end=14, screenshot_time=40)], 60.0)
    assert a['screenshot_time'] == 12.0


# ---------- ffmpeg ----------

def test_cut_clip_is_frame_accurate_not_keyframe_snapped(av_clip, tmp_path):
    # The source has a single keyframe at 0s; a stream copy from 1.3s would start on red. Re-encoding lands on white.
    cut = download.cut_clip(av_clip, 1.3, 3.0, tmp_path / 'c.mp4')
    r, g, b = _first_pixel(tmp_path / 'c.mp4')
    assert min(r, g, b) > 200, (r, g, b)
    assert abs(cut['duration'] - 1.7) < 0.15 and cut['has_audio']
    video_s, audio_s = download.probe_times(tmp_path / 'c.mp4')
    assert abs(video_s - audio_s) < 0.15  # audio and video stay in sync


def test_cut_clip_without_audio_and_bad_ranges(static_clip, tmp_path):
    assert download.cut_clip(static_clip, 0.5, 2.0, tmp_path / 'n.mp4')['has_audio'] is False
    with pytest.raises(SystemExit, match='start < end'):
        download.cut_clip(static_clip, 2.0, 1.0, tmp_path / 'x.mp4')
    with pytest.raises(SystemExit, match='could not cut'):
        download.cut_clip(tmp_path / 'missing.mp4', 0, 1, tmp_path / 'y.mp4')
    assert not (tmp_path / 'y.mp4').exists()


def test_screenshot_is_full_resolution_png(av_clip, tmp_path):
    download.grab_screenshot(av_clip, 1.0, tmp_path / 's' / 'shot.png')
    data = (tmp_path / 's' / 'shot.png').read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    assert int.from_bytes(data[16:20], 'big') == 320 and int.from_bytes(data[20:24], 'big') == 240  # not downscaled to 512
    with pytest.raises(SystemExit, match='no frame'):
        download.grab_screenshot(av_clip, 999, tmp_path / 'late.png')


# ---------- rendering ----------

SUMMARY = {
    'title': 'Deadlift <b>form</b>',
    'overview': ['It teaches the hinge.', 'It is slow.'],
    'sections': [
        {'heading': 'Setup', 'kind': 'topic', 'blocks': [
            {'kind': 'takeaway', 'sentences': ['Bar over midfoot.', 'Shins vertical.'], 'bullets': ['Grip', 'Brace']},
            {'kind': 'critical', 'sentences': ['Never round the back.']}]},
        {'heading': 'Pull', 'kind': 'bogus', 'blocks': [{'kind': 'technical', 'sentences': ['Hips and shoulders rise together.']},
                                                         {'kind': 'insight', 'bullets': ['Straps help.']}]},
    ],
}
CLIPS = [{'title': 'Setup <script>', 'category': 'Deadlift', 'subcategory': 'Setup', 'start': 1.0, 'end': 5.5,
          'duration': 4.5, 'mp4': 'video_snippets/01_Deadlift/01_Setup/Action_01_Setup/action_01.mp4',
          'info': 'x/info.json', 'description': 'Lifter sets up.', 'screenshot': 'screenshots/important_moment_01.png'}]
META = {'channel': 'Coach', 'upload_date_iso': '2026-01-02', 'view_count': 12345, 'webpage_url': 'https://example.com/v'}


def test_html_enforces_the_requested_spacing_and_colors():
    page = render.render_html(render.clean_summary(SUMMARY), CLIPS, META, SUMMARY['title'])
    assert '--topic-gap:calc(var(--line)*3.5)' in page and '--sentence-gap:var(--line)' in page
    assert 'font:16px/2 ' in page  # line-height 2 = double spacing
    for kind in render.KINDS:
        assert f'--{kind}:' in page and f'.kind-{kind}{{' in page.replace('}.', '}\n.')
    assert '@media (prefers-color-scheme:dark)' in page
    assert page.count('<p class="sentence') == 2 + 2 + 1 + 1 + 1  # overview 2, takeaway 2, critical 1, technical 1, clip line 1
    assert 'kind-takeaway' in page and 'kind-critical' in page and 'kind-technical' in page and 'kind-insight' in page
    assert 'kind-bogus' not in page  # unknown kinds fall back to topic


def test_html_escapes_everything_from_the_video():
    page = render.render_html(render.clean_summary(SUMMARY), CLIPS, META, SUMMARY['title'])
    assert '<script>' not in page and '<b>form</b>' not in page
    assert '&lt;script&gt;' in page
    evil = render.render_html(render.clean_summary(SUMMARY), [], {'webpage_url': 'javascript:alert(1)'}, 't')
    assert 'href="javascript' not in evil


def test_markdown_double_spaces_sentences_and_marks_colors():
    md = render.render_markdown(render.clean_summary(SUMMARY), CLIPS, META, 'T')
    assert '🟢 Bar over midfoot.\n\n🟢 Shins vertical.\n\n' in md
    for marker in ('🔵', '🟢', '🔴', '🟠', '🟣'):
        assert marker in md
    assert md.count('<br><br><br>') >= 3 and 'video_summary.html' in md


def test_empty_summary_is_refused():
    with pytest.raises(SystemExit, match='at least one section'):
        render.clean_summary({'sections': [{'heading': 'x', 'blocks': []}]})


# ---------- package build ----------

ANALYSIS = {
    'summary': SUMMARY,
    'clips': [
        _clip('Setup stance', 1.0, 2.2, category='Deadlift', subcategory='Setup', importance=4),
        _clip('Setup stance again', 1.2, 2.4, category='Deadlift', subcategory='Setup', importance=2),
        _clip('Hip drive', 3.0, 4.6, category='Deadlift', subcategory='Pull', importance=5),
        _clip('Warm-up', 0.0, 1.0, category='Prep', subcategory='', importance=3),
        _clip('Too late', 80, 90),
    ],
    'moments': [{'title': 'Title card', 'time': 5.0}, {'title': 'Beyond', 'time': 500}],
}


def _build(av_clip, root, analysis=ANALYSIS, **kwargs):
    return package.build_package(analysis, str(av_clip), root, metadata={'id': 'abc', 'title': 'Deadlift guide', **META},
                                 source={'title': 'Deadlift guide', 'url': 'https://example.com/v'}, **kwargs)


def test_build_package_writes_the_required_tree(av_clip, tmp_path):
    root = tmp_path / 'video_analysis'
    report = _build(av_clip, root)
    assert [c['title'] for c in report['clips']] == ['Warm-up', 'Setup stance', 'Hip drive']
    assert {r['title'] for r in report['rejected']} == {'Setup stance again', 'Too late'}
    assert not report['failures']
    for rel in ('summaries/video_summary.md', 'summaries/video_summary.html', 'metadata/video_metadata.json',
                'index.html', 'index.json', 'screenshots/important_moment_01.png', 'screenshots/important_moment_04.png'):
        assert (root / rel).is_file(), rel
    assert not (root / 'screenshots' / 'important_moment_05.png').exists()  # the out-of-range moment was dropped
    mp4s = sorted(p.relative_to(root).as_posix() for p in root.glob('video_snippets/**/*.mp4'))
    assert mp4s == [
        'video_snippets/01_Prep/01_General/Action_01_Warm_up/action_01.mp4',
        'video_snippets/02_Deadlift/01_Setup/Action_02_Setup_stance/action_02.mp4',
        'video_snippets/02_Deadlift/02_Pull/Action_03_Hip_drive/action_03.mp4']
    info = json.loads((root / 'video_snippets/02_Deadlift/02_Pull/Action_03_Hip_drive/action_03_info.json').read_text())
    assert info['clip_title'] == 'Hip drive' and info['category'] == 'Deadlift' and info['subcategory'] == 'Pull'
    assert info['action_description'] == 'Lifter squats.'
    assert (info['action_start_seconds'], info['action_end_seconds']) == (3.0, 4.6)
    assert (info['start_seconds'], info['end_seconds']) == (2.0, 5.6)  # one second of context each side, clamped at 5.6
    assert info['start_timestamp'] == '00:02.0' and info['end_timestamp'] == '00:05.6'
    assert abs(info['duration_seconds'] - 3.6) < 0.15
    assert info['source_video'] == {'title': 'Deadlift guide', 'url': 'https://example.com/v'}
    assert (root / info['related_screenshot']).is_file()
    index = json.loads((root / 'index.json').read_text())
    assert [c['number'] for c in index['clips']] == [1, 2, 3] and len(index['rejected_clips']) == 2
    meta = json.loads((root / 'metadata/video_metadata.json').read_text())
    assert meta['channel'] == 'Coach' and meta['video_file']['width'] == 320
    html_index = (root / 'index.html').read_text()
    assert 'Action_03_Hip_drive/action_03.mp4' in html_index and '00:02.0 → 00:05.6' in html_index


def test_rerun_replaces_stale_clips_but_a_different_video_is_refused(av_clip, tmp_path):
    root = tmp_path / 'out'
    _build(av_clip, root)
    smaller = {**ANALYSIS, 'clips': [_clip('Only', 1, 3, category='One', subcategory='A')], 'moments': []}
    _build(av_clip, root, smaller)
    assert len(list(root.glob('video_snippets/**/*.mp4'))) == 1 and len(list(root.glob('screenshots/*.png'))) == 1
    with pytest.raises(SystemExit, match='different video'):
        package.build_package(smaller, str(av_clip), root, metadata={'id': 'other'}, source={'title': 'o', 'url': 'u'})


def test_foreign_non_empty_directory_is_never_touched(av_clip, tmp_path):
    root = tmp_path / 'docs'
    root.mkdir()
    (root / 'keep.txt').write_text('mine')
    with pytest.raises(SystemExit, match='not empty'):
        _build(av_clip, root)
    assert (root / 'keep.txt').read_text() == 'mine'


def test_one_failed_cut_is_reported_and_the_rest_continue(av_clip, tmp_path):
    def flaky(video, start, end, dest):
        if start < 1.5:
            raise SystemExit('ffmpeg exploded')
        return download.cut_clip(video, start, end, dest)
    report = _build(av_clip, tmp_path / 'p', cutter=flaky)
    assert len(report['failures']) == 2 and len(report['clips']) == 1
    assert 'exploded' in report['failures'][0]['error']
    assert json.loads((tmp_path / 'p/index.json').read_text())['failures']


def test_package_cli_end_to_end(av_clip, tmp_path):
    analysis = tmp_path / 'analysis.json'
    analysis.write_text(json.dumps(ANALYSIS), encoding='utf-8')
    root = tmp_path / 'cli out'
    result = subprocess.run([sys.executable, str(SCRIPT), str(av_clip), '--package', str(analysis),
                             '--analysis-dir', str(root), '--clip-pad', '0.5'], capture_output=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
    assert '# watch: analysis package' in result.stdout and 'Rejected clips' in result.stdout
    assert (root / 'index.html').is_file()
    info = json.loads(next(root.glob('**/action_03_info.json')).read_text())
    assert info['context_padding_seconds'] == 0.5


def test_package_rejects_a_malformed_analysis(tmp_path):
    bad = tmp_path / 'a.json'
    bad.write_text('{"clips": []}')
    with pytest.raises(SystemExit, match='"summary"'):
        package.load_analysis(bad)
    bad.write_text('not json')
    with pytest.raises(SystemExit, match='Cannot read'):
        package.load_analysis(bad)


# ---------- metadata, quality, cache ----------

def test_rich_metadata_keeps_present_fields_only():
    meta = download.rich_metadata({'id': 'x', 'title': 'T', 'upload_date': '20260102', 'view_count': 5, 'tags': [],
                                   'description': '', 'channel': 'C', 'categories': ['Sports'], 'junk': 1}, 'https://e/x')
    assert meta == {'id': 'x', 'title': 'T', 'upload_date': '20260102', 'view_count': 5, 'channel': 'C',
                    'categories': ['Sports'], 'webpage_url': 'https://e/x', 'upload_date_iso': '2026-01-02'}


def _fake_ytdlp(monkeypatch, calls):
    def run(cmd):
        calls.append(cmd)
        directory = Path(cmd[cmd.index('-o') + 1].rsplit('/video.', 1)[0].replace('%%', '%'))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'video.info.json').write_text(json.dumps({'id': 'vid1', 'title': 'Cached', 'view_count': 9}))
        path = directory / 'video.mp4'
        path.write_bytes(b'media')
        Path(cmd[cmd.index('--print-to-file') + 2].replace('%%', '%')).write_text(json.dumps(str(path)) + '\n')
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(download, '_run', run)


def test_quality_selects_the_format(monkeypatch, tmp_path):
    calls = []
    _fake_ytdlp(monkeypatch, calls)
    download.download_url('https://e.com/v', tmp_path)
    download.download_url('https://e.com/v', tmp_path, quality='best')
    assert 'height<=720' in calls[0][calls[0].index('-f') + 1]
    assert calls[1][calls[1].index('-f') + 1] == 'bv*+ba/b'
    with pytest.raises(SystemExit, match='--quality'):
        download.download_url('https://e.com/v', tmp_path, quality='ultra')


def test_cache_downloads_once_and_carries_metadata(monkeypatch, tmp_path):
    calls = []
    _fake_ytdlp(monkeypatch, calls)
    cache = tmp_path / 'cache'
    first = download.download('https://e.com/v', tmp_path / 'w1', cache_dir=cache, quality='best')
    second = download.download('https://e.com/v', tmp_path / 'w2', cache_dir=cache, quality='best')
    assert len(calls) == 1 and second['cached'] and not first['cached']
    assert second['video_path'] == first['video_path'] and Path(second['video_path']).read_bytes() == b'media'
    assert second['metadata']['view_count'] == 9 and second['info']['title'] == 'Cached'
    download.download('https://e.com/v', tmp_path / 'w3', cache_dir=cache, quality='default')
    assert len(calls) == 2  # a different quality is a different entry


def test_cache_ignores_missing_or_empty_files_and_skips_audio_only(monkeypatch, tmp_path):
    calls = []
    _fake_ytdlp(monkeypatch, calls)
    cache = tmp_path / 'cache'
    hit = download.download('https://e.com/v', tmp_path / 'w1', cache_dir=cache)
    Path(hit['video_path']).write_bytes(b'')
    assert download.cache_lookup('https://e.com/v', 'default', cache) is None
    Path(hit['video_path']).unlink()
    assert download.cache_lookup('https://e.com/v', 'default', cache) is None
    download.download('https://e.com/v', tmp_path / 'w2', audio_only=True, cache_dir=cache)
    assert not any(p.name == 'media.json' and 'audio' in p.read_text() for p in cache.glob('*/media.json'))


# ---------- watch.py flags and gemini analysis mode ----------

def test_uniform_frames_for_pinning_an_action_window(av_clip, tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), str(av_clip), '--no-whisper', '--uniform', '--start', '1', '--end', '2',
                             '--out-dir', str(tmp_path / 'o')], capture_output=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
    assert 'uniform' in result.stdout and result.stdout.count('reason=uniform') >= 2
    assert '(t=00:01' in result.stdout


def test_analyze_prints_the_workflow_and_schema(av_clip, tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), str(av_clip), '--no-whisper', '--analyze',
                             '--analysis-dir', str(tmp_path / 'a'), '--out-dir', str(tmp_path / 'o')],
                            capture_output=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
    assert '## Analysis workflow' in result.stdout and '"clips"' in result.stdout and '--package' in result.stdout
    assert '## Video metadata' not in result.stdout  # a local file has none; no empty heading


def test_gemini_analyze_rejects_a_focus_range(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), 'https://youtu.be/abc', '--engine', 'gemini', '--analyze', '--start', '5'],
                            capture_output=True, encoding='utf-8', env={'GEMINI_API_KEY': 'k', 'PATH': '/usr/bin:/bin'})
    assert result.returncode == 2 and 'source timeline' in result.stderr


def test_gemini_analysis_prompt_carries_the_schema_and_demands_json():
    prompt = gemini.build_prompt('focus on footwork', analysis=True)
    assert 'footwork' in prompt and '"clips"' in prompt and 'ONLY one JSON object' in prompt
    assert 'Separate what is seen' not in prompt  # the prose-answer suffix does not apply
    assert gemini.build_prompt('q') == 'q' + gemini.PROMPT_SUFFIX  # normal mode unchanged
