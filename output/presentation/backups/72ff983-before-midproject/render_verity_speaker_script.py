"""Render matching five-speaker Markdown from the canonical slide content."""
from pathlib import Path
from collections import defaultdict
import json

ROOT=Path(__file__).resolve().parents[1]


def render():
    data=json.loads((ROOT/'docs/presentation/verity-technical-content.json').read_text(encoding='utf-8'))
    slides=data['slides']
    assert [r['n'] for r in slides]==list(range(1,33))
    grouped=defaultdict(list)
    for row in slides:
        if not row['appendix']:grouped[row['speaker']].append(row)
    out=['# Verity technical presentation: five-speaker script','',
         'Current version: 29 core slides and three optional technical backup slides. Allow approximately 24–26 minutes including transitions. Read the quoted paragraphs aloud. Technical detail and evidence are for rehearsal and Q&A, not additional required speech. Speaking roles need not match implementation ownership. Replace Member 1–5 with names only after agreeing the speaking order.','',
         'The deployed checkpoint is `YukinoKaren/ReasonKE_plus`, used for the Reason-KE++ method. Runtime corrections change approved external memory while its weights stay frozen. We do not claim new SFT/PPO training or reproduction of the paper’s headline score. The single concise oral attribution is on slide 5. References stay in notes and this script rather than crowding slide faces.','',
         '## Speaking order','',
         '| Speaker | Core slides | Spoken words | Speech at 125 words/minute |',
         '| --- | --- | ---: | ---: |']
    for member,rows in sorted(grouped.items()):
        words=sum(len(r['speech'].split()) for r in rows)
        out.append(f'| Member {member} | {rows[0]["n"]}–{rows[-1]["n"]} | {words} | {words/125:.1f} minutes |')
    for member,rows in sorted(grouped.items()):
        out+=['',f'## Member {member}','']
        for r in rows:
            out+=section(r)
        if member<5:out+=['',f'**Handover:** “Member {member+1} will take the next part.”','']
    out+=['','## Optional technical backup slides','',
          'Use these only when the panel asks. They are outside the core speaking-time allocation.','']
    for r in slides:
        if r['appendix']:out+=section(r)
    out+=['','## Technical questions and defensible answers','']
    for qa in data['qa']:out+=[f'### {qa["question"]}','',qa['answer'],'']
    out+=['## Demonstration and fallback','',
          'Use the current [India demonstration guide](../INDIA_DEMO.md). Begin with a ready model worker and a fresh demo workspace. Ask the three-hop office question, approve the ownership replacement, request a fresh answer, check unchanged hosting and historical-release facts, then roll back and request another fresh answer. A live comparison makes two model calls and takes longer than one question.','',
          'If the GPU worker is unavailable, identify the saved responses as recorded evidence. The screenshot workspace is a replay of actual audited responses and final state, not a deterministic mock or new inference. Do not click a recorded chat and describe it as a fresh answer. Do not replace a failed live model answer with the expected database target.','',
          '## Evidence-based contribution allocation','',
          'These areas cover implemented, reviewable work. They are an allocation map, not invented biographical history. Before attaching names, each person must confirm their actual implementation, reproduction, review, analysis or documentation. Several people may collaborate on one area. Presentation assignments are independent.','',
          '| Member | Contribution area | Concrete work to describe once confirmed | Evidence |',
          '| --- | --- | --- | --- |']
    for row in data['contributions']:
        out.append('| '+' | '.join(row[k] for k in ('member','area','work','evidence'))+' |')
    target=ROOT/'docs/presentation/VERITY_SPEAKER_SCRIPT.md'
    target.write_text('\n'.join(out)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'core_slides':29,'backup_slides':3,'core_spoken_words':sum(len(r['speech'].split()) for r in slides if not r['appendix']),'speakers':5,'script':str(target)},indent=2))


def section(row):
    sources=[]
    for source in row['sources']:
        target=source if source.startswith('https:') else '../../'+source
        sources.append(f'- [{source}]({target})')
    return [f'### Slide {row["n"]}: {row["title"]}','',
            f'**Visual cue:** {row["cue"]}','',
            '> '+row['speech'],'',
            '**Technical detail for rehearsal:** '+row['detail'],'',
            '**Evidence:**','',*sources,'']


if __name__=='__main__':render()
