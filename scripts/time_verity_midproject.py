"""Apply the declared 600-second native PowerPoint timings without changing content."""
import argparse
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
P = '{http://schemas.openxmlformats.org/presentationml/2006/main}'


def apply(source, target):
    source, target = Path(source), Path(target)
    if target.exists() or source.resolve() == target.resolve():
        raise ValueError('Choose a new output; preserve the source package')
    rows = json.loads((ROOT/'docs/presentation/verity-midproject-content.json').read_text(encoding='utf-8'))['slides']
    assert len(rows) == 13 and sum(r['seconds'] for r in rows) == 600
    with zipfile.ZipFile(source) as zin, zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            match = re.fullmatch(r'ppt/slides/slide(\d+)\.xml', item.filename)
            if match:
                n = int(match[1]); xml = data.decode('utf-8')
                if ET.fromstring(xml).find(P+'transition') is not None:
                    raise ValueError('Unexpected existing transition: preserve rather than silently replace it')
                assert xml.endswith('</p:sld>')
                transition = f'<p:transition advClick="0" advTm="{rows[n-1]["seconds"]*1000}" />'
                data = (xml[:-len('</p:sld>')]+transition+'</p:sld>').encode('utf-8')
            elif item.filename == 'ppt/presProps.xml':
                xml = data.decode('utf-8')
                assert ET.fromstring(xml).find(P+'showPr') is None
                xml = xml.replace('<p:extLst>', '<p:showPr useTimings="1" /><p:extLst>', 1)
                assert ET.fromstring(xml).find(P+'showPr').get('useTimings') == '1'
                data = xml.encode('utf-8')
            zout.writestr(item, data)
    with zipfile.ZipFile(target) as z:
        ms = [int(ET.fromstring(z.read(f'ppt/slides/slide{r["n"]}.xml')).find(P+'transition').get('advTm')) for r in rows]
        assert sum(ms) == 600000
    print(json.dumps({'slides': len(rows), 'milliseconds': sum(ms), 'output': str(target)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    apply(args.source, args.target)
