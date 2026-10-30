"""Restore public recorded website output in a fresh screenshot-only workspace.

This makes zero model calls. Response text, timestamps, revisions and generation
provenance come from the published India receipt. Chat titles identify the replay.
It does not add observations or change any released evidence.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json

from editguard_chat.store import ChatStore, correction_identity, _json
from editguard_chat.workspace import COLUMNS, seal
from scripts.audit_india_http import audit

ROOT=Path(__file__).resolve().parents[1]


def prepare(output):
    output=Path(output)
    if output.exists():
        raise ValueError('Choose a fresh screenshot workspace; existing data is preserved')
    receipt_path=ROOT/'reports/reasonke/depth-final/india-acceptance.json'
    audit(receipt_path)
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    operations=receipt['operations']
    final=next(o['response'] for o in operations if o['name']=='inspect_restored_nine_facts')
    tables={key:[] for key in COLUMNS}
    for value in final['corrections']:
        row={key:value[key] for key in COLUMNS['corrections'] if key in value}
        key,scope,identity=correction_identity(value['question'],value['scope'],value['identity'])
        row.update(question_key=key,scope_key=scope,identity_json=_json(identity))
        tables['corrections'].append(row)
    for value in final['events']:
        row={key:value[key] for key in COLUMNS['events'] if key in value}
        row['detail_json']=_json(value['detail'])
        tables['events'].append(row)
    for op in operations:
        if not op['name'].startswith('generate_'):
            continue
        response=op['response']
        chat_id=response['user']['chat_id']
        original=next(o['response']['chat'] for o in operations
                      if o['name'].startswith('create_') and o['response']['chat']['id']==chat_id)
        chat={**original,'title':'Recorded: '+op['name'].removeprefix('generate_').replace('_',' '),
              'updated_at':response['assistant']['created_at']}
        tables['chats'].append(chat)
        for role in ('user','assistant'):
            value=response[role]
            row={key:value[key] for key in COLUMNS['messages'] if key in value}
            row.update(evidence_json=_json(value.get('evidence',{})),image_path=None)
            tables['messages'].append(row)
    snapshot=seal({'schema':'editguard-workspace-v1','created_at':datetime.now(timezone.utc).isoformat(),
                   'revision':final['revision'],'tables':tables})
    output.mkdir(parents=True)
    store=ChatStore(output/'chat.sqlite3')
    try:
        store.import_snapshot(snapshot)
        if store.revision!=11:
            raise ValueError('The recorded rollback state must remain revision eleven')
    finally:
        store.close()
    manifest={'schema':'verity-screenshot-replay-v1','source_receipt':str(receipt_path.relative_to(ROOT)),
              'source_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
              'model_calls':0,'recorded_chat_count':len(tables['chats']),
              'recorded_messages':len(tables['messages']),'knowledge_revision':11,
              'notice':'Actual saved model responses displayed in the current interface. Not new inference or benchmark observations. Live readiness must remain truthful.'}
    (output/'replay-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)
