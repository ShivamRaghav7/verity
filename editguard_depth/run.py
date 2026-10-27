"""Finite depth stress study, isolated from the completed operations experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import re
import time

from editguard_benchmark.run import _worker
from .retrieval import retrieve_iterative
from editguard_benchmark.scoring import score
from editguard_benchmark.data import digest, encoded
from editguard_reasonke.__main__ import write_receipt
from .data import load_dataset

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def identities(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    paths = sorted((ROOT / 'editguard_depth').glob('*.py'))
    paths += sorted((ROOT / 'editguard_benchmark').glob('*.py'))
    paths += [ROOT / 'editguard_reasonke/runtime.py', config_path]
    paths += [ROOT / config[key] for key in ('reasonke_manifest', 'instruction_manifest')]
    paths += [ROOT / config['dataset'] / 'manifest.json']
    if config.get('retrieval_selection'):
        paths.append(ROOT / config['retrieval_selection'])
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}


def tasks_for(config, model, phase):
    manifest, worlds, questions = load_dataset(ROOT / config['dataset'], 'final' if phase == 'final' else 'development')
    if phase != 'final':
        worlds_selected = sorted(worlds)[:1 if phase == 'profile' else 3]
        questions = [row for row in questions if row['lineage_id'] in worlds_selected]
    prefix = 'R' if model == 'reasonke' else 'G'
    tasks = [{'id': prefix + '-RAG:' + q['id'], 'condition': prefix + '-RAG', 'question': q, 'outcome': 'unexecuted'}
             for q in sorted(questions, key=lambda q: q['id'])]
    if phase == 'final':
        selected = sorted((q for q in questions if q['depth'] == 3),
                          key=lambda q: hashlib.sha256(('depth-oracle-43:' + q['id']).encode()).hexdigest())[:16]
        tasks += [{'id': prefix + '-ORACLE:' + q['id'], 'condition': prefix + '-ORACLE', 'question': q,
                   'outcome': 'unexecuted'} for q in selected]
    return manifest, worlds, tasks


def request_for(task, worlds, config):
    question = task['question']
    facts = worlds[question['lineage_id']]['current_facts']
    if task['condition'].endswith('ORACLE'):
        facts = [fact for fact in facts if fact['id'] in question['required_fact_ids']]
    else:
        facts = retrieve_iterative(question['question'], facts, config['retrieval_k'])
    required = set(question['required_fact_ids'])
    selected = [fact['id'] for fact in facts]
    return {'question': question['question'], 'history': [], 'max_new_tokens': config['max_new_tokens'],
            'facts': [{key: fact[key] for key in ('id', 'question', 'answer', 'scope', 'source')} for fact in facts]}, {
                'required': len(required), 'found': len(required.intersection(selected)),
                'complete': required.issubset(selected), 'selected_fact_ids': selected,
                'gold_support_used': task['condition'].endswith('ORACLE')}


def verify_protocol(path, config_path):
    protocol = json.loads(Path(path).read_text())
    if (protocol.get('schema') != 'editguard-depth-protocol-v1' or protocol.get('status') != 'frozen'
            or protocol.get('sources') != identities(config_path)
            or protocol.get('final_assigned') != 544 or protocol.get('per_main_condition') != 256
            or protocol.get('oracle_per_model') != 16):
        raise ValueError('Depth source/data/configuration differs from frozen protocol')
    profiles = protocol.get('profiles', {})
    inputs = protocol.get('input_receipts', {})
    expected = {f'{phase}-{model}.json' for phase in ('profile', 'development') for model in ('reasonke', 'instruction')}
    if set(profiles) != {'reasonke', 'instruction'} or set(inputs) != expected:
        raise ValueError('Both depth profiles and development receipts are required')
    for profile in profiles.values():
        if (profile.get('n') != 8 or profile.get('final_assigned') != 272
                or not all(isinstance(profile.get(key), (int, float)) and math.isfinite(profile[key]) and profile[key] > 0
                           for key in ('mean_seconds', 'max_seconds', 'forecast_seconds_mean', 'forecast_seconds_max'))
                or profile['max_seconds'] < profile['mean_seconds']
                or not math.isclose(profile['forecast_seconds_mean'], 272*profile['mean_seconds'])
                or not math.isclose(profile['forecast_seconds_max'], 272*profile['max_seconds'])):
            raise ValueError('Depth measured profile budget is invalid')
    from .metrics import audit_and_aggregate
    for name, checksum in inputs.items():
        receipt_path = Path(path).parent / name
        if not re.fullmatch('[a-f0-9]{64}', checksum) or not receipt_path.exists() or digest(receipt_path) != checksum:
            raise ValueError('Depth frozen input receipt missing or changed: '+name)
        receipt = json.loads(receipt_path.read_text())
        phase, model = name.removesuffix('.json').split('-', 1)
        if (receipt['status'] != 'completed' or receipt['plan']['model'] != model
                or receipt['plan']['phase'] != phase or len(receipt['records']) != (8 if phase == 'profile' else 24)
                or any(row['outcome'] == 'unexecuted' for row in receipt['records'])):
            raise ValueError('Depth frozen phase population is incomplete or mismatched')
        audit_and_aggregate([receipt_path], config_path=config_path)
        if phase == 'profile':
            times = [row['pipeline_seconds'] for row in receipt['records'] if row.get('response')]
            if (len(times) != 8 or not math.isclose(sum(times)/8, profiles[model]['mean_seconds'])
                    or not math.isclose(max(times), profiles[model]['max_seconds'])):
                raise ValueError('Depth budget does not match measured profile timings')
    return protocol


def freeze(directory, output, config_path):
    directory, output = Path(directory), Path(output)
    if output.resolve().parent != directory.resolve():
        raise ValueError('Keep the depth protocol beside its four frozen input receipts')
    if output.exists():
        raise ValueError('Preserve existing depth protocol')
    sources = identities(config_path)
    inputs, profiles = {}, {}
    for model in ('reasonke', 'instruction'):
        for phase, expected in (('profile', 8), ('development', 24)):
            path = directory / f'{phase}-{model}.json'
            receipt = json.loads(path.read_text())
            if (receipt['status'] != 'completed' or receipt['plan']['sources'] != sources
                    or receipt['plan']['model'] != model or receipt['plan']['phase'] != phase
                    or len(receipt['records']) != expected or any(r['outcome'] == 'unexecuted' for r in receipt['records'])):
                raise ValueError('Complete declared depth profile/development inputs required')
            from .metrics import audit_and_aggregate
            audit_and_aggregate([path], config_path=config_path)
            inputs[path.name] = digest(path)
            if phase == 'profile':
                timings = [r['pipeline_seconds'] for r in receipt['records'] if r.get('response')]
                if (len(timings) != expected or any(not isinstance(t, (int, float))
                        or not math.isfinite(t) or t <= 0 for t in timings)):
                    raise ValueError('All profile generations required for measured budget')
                profiles[model] = {'n': expected, 'mean_seconds': sum(timings) / len(timings),
                                   'max_seconds': max(timings), 'final_assigned': 272,
                                   'forecast_seconds_mean': 272 * sum(timings) / len(timings),
                                   'forecast_seconds_max': 272 * max(timings),
                                   'limitation': 'Eight development requests; uncertain extrapolation excluding fresh loading/downloads'}
    protocol = {'schema': 'editguard-depth-protocol-v1', 'status': 'frozen', 'created_utc': now(),
                'sources': sources, 'input_receipts': inputs, 'profiles': profiles, 'final_assigned': 544,
                'per_main_condition': 256, 'oracle_per_model': 16,
                'question': 'Do the development-selected iterative retrieval policy and each checkpoint support three-hop edits on new graphs and held-out wording families?',
                'policy': 'Iterative answer-frontier retrieval: four facts, two lexical seeds, three expansion rounds selected on development; shared across models. Matched model budgets and fixed whole-answer scoring; no post-test tuning. Completed operations-v1 is unchanged.',
                'review': 'Automated controlled graph audit; no independent human semantic review',
                'not_combined_with': 'Completed operations-v1 scorecard'}
    write_receipt(output, protocol)
    return protocol


def run(config_path, model, phase, output, protocol=None, resume=False):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if phase == 'final':
        if protocol is None:
            raise ValueError('Frozen depth protocol required before accessing final questions')
        verify_protocol(protocol, config_path)
    manifest, worlds, tasks = tasks_for(config, model, phase)
    plan = {'model': model, 'phase': phase, 'sources': identities(config_path),
            'task_ids_sha256': hashlib.sha256(encoded([r['id'] for r in tasks])).hexdigest(),
            'protocol_sha256': digest(protocol) if protocol else None}
    output = Path(output)
    if output.exists():
        if not resume:
            raise ValueError('Use exact-plan --resume; existing outcomes are never overwritten')
        receipt = json.loads(output.read_text())
        if receipt.get('plan') != plan or [r['id'] for r in receipt['records']] != [r['id'] for r in tasks]:
            raise ValueError('Depth resume plan changed')
        from .metrics import audit_and_aggregate
        audit_and_aggregate([output], config_path=config_path, protocol=protocol)
    else:
        receipt = {'schema': 'editguard-depth-inference-v1', 'status': 'running', 'started_utc': now(),
                   'plan': plan, 'dataset': manifest, 'records': tasks, 'sessions': [],
                   'notice': 'Separate depth challenge. All outcomes retained. No paper scores, answer retries or training.'}
    remaining = [r for r in receipt['records'] if r['outcome'] == 'unexecuted']
    if not remaining:
        return receipt
    receipt['status'] = 'running'
    session = {'started_utc': now(), 'remaining_at_start': len(remaining)}
    receipt['sessions'].append(session)
    write_receipt(output, receipt)
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(target=_worker, args=(child, model, config), daemon=True)
    start = time.perf_counter()
    process.start(); child.close()
    active_row = None
    try:
        if not parent.poll(config['model_load_watchdog_seconds']):
            raise TimeoutError('Depth model-load watchdog reached')
        ready = parent.recv()
        if not ready.get('ready'):
            raise RuntimeError('Depth model load failed: ' + json.dumps(ready))
        session.update(runtime=ready['runtime'], load_wall_seconds=time.perf_counter() - start)
        write_receipt(output, receipt)
        for row in remaining:
            if time.perf_counter() - start > config['final_wall_budget_seconds_per_model']:
                raise TimeoutError('Finite depth per-model budget reached')
            began = time.perf_counter()
            payload, routing = request_for(row, worlds, config)
            row.update(started_utc=now(), supplied_facts=payload['facts'], retrieval=routing,
                       request_sha256=hashlib.sha256(encoded(payload)).hexdigest())
            active_row = row
            parent.send(payload)
            if not parent.poll(config['request_watchdog_seconds']):
                row.update(error={'type': 'TimeoutError', 'message': 'Hard depth request watchdog'},
                           pipeline_seconds=time.perf_counter() - began)
                row.update(score(row['question'], error=row['error']));write_receipt(output, receipt)
                raise TimeoutError('Depth request deadline; no automatic outcome retry')
            row.update(parent.recv(), pipeline_seconds=time.perf_counter() - began, completed_utc=now())
            row.update(score(row['question'], row.get('response'), row.get('error')))
            write_receipt(output, receipt)
            print(json.dumps({'id': row['id'], 'outcome': row['outcome'], 'seconds': round(row['pipeline_seconds'], 3)}), flush=True)
        receipt['status'] = 'completed'
    except BaseException as exc:
        # A sent request with no returned verdict is an execution failure. Keep
        # it in the denominator instead of rerunning it on exact-plan resume.
        if active_row is not None and active_row['outcome'] == 'unexecuted':
            active_row.update(error={'type': type(exc).__name__, 'message': str(exc)},
                              pipeline_seconds=time.perf_counter() - began, completed_utc=now())
            active_row.update(score(active_row['question'], error=active_row['error']))
        receipt.update(status='interrupted', last_error={'type': type(exc).__name__, 'message': str(exc)})
    finally:
        if process.is_alive():
            try:
                parent.send(None);process.join(timeout=3)
            except (OSError, EOFError):
                pass
            if process.is_alive():
                process.terminate();process.join(timeout=10)
        parent.close()
        session.update(completed_utc=now(), wall_seconds=time.perf_counter() - start)
        receipt.update(recorded=sum(r['outcome'] != 'unexecuted' for r in receipt['records']),
                       unexecuted=sum(r['outcome'] == 'unexecuted' for r in receipt['records']))
        write_receipt(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/depth-evaluation.json')
    parser.add_argument('--model', choices=('reasonke', 'instruction'))
    parser.add_argument('--phase', choices=('profile', 'development', 'final'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--freeze-directory', type=Path)
    args = parser.parse_args()
    if args.freeze_directory:
        print(json.dumps(freeze(args.freeze_directory, args.output, args.config), indent=2))
    else:
        if not args.model or not args.phase:
            parser.error('--model and --phase required')
        result = run(args.config, args.model, args.phase, args.output, args.protocol, args.resume)
        print(json.dumps({'status': result['status'], 'recorded': result['recorded'] if 'recorded' in result else len(result['records'])}))
        if result['status'] != 'completed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
