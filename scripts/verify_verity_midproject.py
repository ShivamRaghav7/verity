"""Audit the timed deck against its sources, backups and recorded demonstration."""
import hashlib
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from editguard_chat.workspace import validate_snapshot

ROOT = Path(__file__).resolve().parents[1]
P = '{http://schemas.openxmlformats.org/presentationml/2006/main}'
C = '{http://schemas.openxmlformats.org/drawingml/2006/chart}'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    content = json.loads((ROOT/'docs/presentation/verity-midproject-content.json').read_text(encoding='utf-8'))
    rows = content['slides']
    assert len(rows) == 13 and sum(r['seconds'] for r in rows) == 600
    assert all(sum(r['seconds'] for r in rows if r['speaker'] == s) == 120 for s in range(1, 6))
    script = (ROOT/'docs/presentation/VERITY_SPEAKER_SCRIPT.md').read_text(encoding='utf-8')
    assert all(r['speech'] in script for r in rows)
    deck = ROOT/'output/presentation/Verity_Mid_Project_Presentation.pptx'
    with zipfile.ZipFile(deck) as z:
        slide_names = [n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)]
        assert len(slide_names) == 13
        assert ET.fromstring(z.read('ppt/presProps.xml')).find(P+'showPr').get('useTimings') == '1'
        images = {hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('ppt/media/')}
        for row in rows:
            n = row['n']; slide = ET.fromstring(z.read(f'ppt/slides/slide{n}.xml'))
            transition = slide.find(P+'transition')
            assert int(transition.get('advTm')) == row['seconds']*1000
            assert transition.get('advClick') == '0'
            notes = ''.join(ET.fromstring(z.read(f'ppt/notesSlides/notesSlide{n}.xml')).itertext())
            assert row['speech'] in notes, f'Notes mismatch slide {n}'
        chart_values = []
        for name in sorted(n for n in z.namelist() if re.fullmatch(r'ppt/slides/charts/chart\d+\.xml', n)):
            xml = ET.fromstring(z.read(name))
            chart_values.append([float(v.text) for v in xml.findall('.//'+C+'val/'+C+'numRef/'+C+'numCache/'+C+'pt/'+C+'v')])
        assert chart_values[0] == [256, 256, 241, 248]
        assert len(chart_values) == 2 and len(chart_values[1]) == 2
        depth = json.loads((ROOT/'reports/reasonke/depth-final/results.json').read_text(encoding='utf-8'))
        expected = [depth['conditions'][c]['overall']['generated_latency']['p50_seconds'] for c in ('R-RAG','G-RAG')]
        assert all(abs(a-b) < 1e-9 for a,b in zip(chart_values[1], expected))
        assert sum(n.endswith('.xlsx') for n in z.namelist()) == 2
        for f in (ROOT/'output/presentation/midproject/screenshots').glob('*.jpg'):
            assert sha(f) in images, f'Capture bytes changed: {f}'
    backup_dir = ROOT/'output/presentation/backups/72ff983-before-midproject'
    backup = json.loads((backup_dir/'backup-manifest.json').read_text(encoding='utf-8'))
    for row in backup['files']:
        assert sha(backup_dir/row['backup_file']) == row['sha256']
    previous = json.loads((ROOT/'output/presentation/verity-technical-manifest.json').read_text(encoding='utf-8'))
    newline_only = []
    for name, digest in previous['evidence_sources'].items():
        if sha(ROOT/name) != digest:
            assert name in ('docs/CONTRIBUTIONS_AND_EVIDENCE.md', 'docs/CURRENT_ARCHITECTURE.md')
            assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == digest, f'Released evidence changed: {name}'
            newline_only.append(name)
    sources = {}
    for study in ('operations-final', 'depth-final'):
        protocol = json.loads((ROOT/f'reports/reasonke/{study}/protocol.json').read_text(encoding='utf-8'))
        for name, digest in protocol['sources'].items():
            assert sha(ROOT/name) == digest, f'Frozen source changed: {name}'
            sources[name] = digest
    demo = json.loads((ROOT/'output/presentation/midproject/air-india-demo-receipt.json').read_text(encoding='utf-8'))
    assert demo['fixture_sha256'] == sha(ROOT/'data/demos/air-india-handover.json')
    assert demo['status'] == 'completed' and demo['completed_model_calls'] == len(demo['cases']) == 7
    assert demo['benchmark_observations'] == 0
    assert sum(c['final_span_matches_target'] for c in demo['cases']) == demo['target_matches'] == 7
    assert all(c['response']['assistant']['evidence']['generation_kind'] == 'model' for c in demo['cases'])
    assert demo['operations'][1]['http_status'] == 409 and demo['operations'][1]['revision_unchanged']
    assert demo['operations'][2]['restored_owned_predecessor']
    for snapshot in demo['snapshots'].values():
        validate_snapshot(snapshot)
    return {'status': 'passed', 'slides': 13, 'seconds': 600, 'speakers': 5,
            'notes_match_script': True, 'embedded_screenshots_match_original_bytes': True,
            'chart_values_match_results': True, 'verified_backup_files': len(backup['files']),
            'unchanged_released_evidence_files': len(previous['evidence_sources']),
            'documentation_with_checkout_newlines_normalized_for_hash': newline_only,
            'frozen_source_files_verified': len(sources), 'demo_generations': 7,
            'demo_target_matches': 7, 'validated_demo_snapshots': len(demo['snapshots']),
            'new_benchmark_observations': 0, 'pptx_sha256': sha(deck)}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
