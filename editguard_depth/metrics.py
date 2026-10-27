"""Separate, fixed-denominator depth study statistics; never merges the old study."""
import argparse
import json
import math
from pathlib import Path

from editguard_benchmark.data import digest, encoded
from editguard_benchmark.metrics import summarize, paired_bootstrap
from editguard_benchmark.scoring import score
from editguard_reasonke.runtime import extract_answer
from editguard_reasonke.__main__ import write_receipt
from .run import ROOT, identities, tasks_for, request_for, verify_protocol


def audit_and_aggregate(paths, config_path=None, protocol=None):
    config_path = Path(config_path or ROOT/'configs/depth-evaluation.json').resolve()
    config = json.loads(config_path.read_text())
    sources = identities(config_path)
    receipts, all_rows, checks = [], [], []
    for path in map(Path, paths):
        receipt = json.loads(path.read_text())
        if receipt.get('schema') != 'editguard-depth-inference-v1' or receipt['plan']['sources'] != sources:
            raise ValueError('Depth receipt source identity differs')
        if receipt.get('status') == 'completed' and any(row['outcome'] == 'unexecuted' for row in receipt['records']):
            raise ValueError('Completed depth receipt contains unexecuted assignments')
        plan = receipt['plan']
        if plan.get('model') not in ('reasonke', 'instruction') or plan.get('phase') not in ('profile', 'development', 'final'):
            raise ValueError('Unknown depth model or phase')
        if receipt.get('status') not in ('running', 'interrupted', 'completed'):
            raise ValueError('Unknown depth execution status')
        if plan['phase'] == 'final':
            if protocol is None:
                raise ValueError('Depth final audit requires frozen protocol')
            verify_protocol(protocol, config_path)
            if plan['protocol_sha256'] != digest(protocol):
                raise ValueError('Depth frozen protocol digest differs')
        manifest, worlds, expected_tasks = tasks_for(config, plan['model'], plan['phase'])
        if receipt['dataset'] != manifest or [r['id'] for r in receipt['records']] != [r['id'] for r in expected_tasks]:
            raise ValueError('Depth task or dataset identity differs')
        import hashlib
        if plan['task_ids_sha256'] != hashlib.sha256(encoded([r['id'] for r in expected_tasks])).hexdigest():
            raise ValueError('Depth assigned task digest differs')
        model = json.loads((ROOT/config[plan['model']+'_manifest']).read_text())
        for row, expected in zip(receipt['records'], expected_tasks):
            if row['question'] != expected['question'] or row['condition'] != expected['condition']:
                raise ValueError('Depth question/condition changed: '+row['id'])
            if row['outcome'] == 'unexecuted':
                if row.get('response') or row.get('error'):
                    raise ValueError('Depth unexecuted record contains a result')
                continue
            request, routing = request_for(expected, worlds, config)
            if (row['request_sha256'] != hashlib.sha256(encoded(request)).hexdigest()
                    or row['supplied_facts'] != request['facts'] or row['retrieval'] != routing):
                raise ValueError('Depth request/routing changed: '+row['id'])
            response = row.get('response')
            latency = row.get('pipeline_seconds')
            if not isinstance(latency, (int, float)) or isinstance(latency, bool) or not math.isfinite(latency) or latency < 0:
                raise ValueError('Depth request latency must be finite and nonnegative: '+row['id'])
            if response:
                content, fmt = extract_answer(response['raw_output'])
                generated = response['generation']
                if (content != response['content'] or fmt != response['format']
                        or generated['model'] != model['model_id'] or generated['revision'] != model['revision']
                        or generated['fact_ids'] != routing['selected_fact_ids']
                        or generated.get('max_new_tokens') != config['max_new_tokens']
                        or generated.get('max_input_tokens') != model['max_input_tokens']
                        or generated.get('do_sample') is not False):
                    raise ValueError('Depth model provenance/extraction differs: '+row['id'])
            scored = score(row['question'], response, row.get('error'))
            if any(row.get(key) != value for key, value in scored.items()):
                raise ValueError('Depth frozen score differs: '+row['id'])
        checks.append({'receipt': str(path), 'sha256': digest(path), 'assigned': len(expected_tasks),
                       'recorded': sum(r['outcome'] != 'unexecuted' for r in receipt['records'])})
        receipts.append(receipt); all_rows += receipt['records']
    if not receipts or len({r['plan']['phase'] for r in receipts}) != 1 or len({r['id'] for r in all_rows}) != len(all_rows):
        raise ValueError('Do not mix phases or duplicate depth outcomes')
    conditions = {}
    for name in sorted({r['condition'] for r in all_rows}):
        rows = [r for r in all_rows if r['condition'] == name]
        condition = {'overall': summarize(rows),
                     'by_depth': {str(depth): summarize([r for r in rows if r['question']['depth'] == depth])
                                  for depth in sorted({r['question']['depth'] for r in rows})},
                     'by_category': {category: summarize([r for r in rows if r['question']['category'] == category])
                                     for category in sorted({r['question']['category'] for r in rows})}}
        required = sum(len(r['question']['required_fact_ids']) for r in rows)
        condition['retrieval'] = {'found': sum(r.get('retrieval', {}).get('found', 0) for r in rows),
                                  'required': required,
                                  'complete': sum(r.get('retrieval', {}).get('complete', False) for r in rows),
                                  'assigned': len(rows), 'oracle': name.endswith('ORACLE')}
        complete = [r for r in rows if r.get('retrieval', {}).get('complete')]
        missing = [r for r in rows if 'retrieval' in r and not r['retrieval']['complete']]
        condition['generation_with_complete_retrieved_support'] = summarize(complete)
        condition['generation_with_missing_retrieved_support'] = summarize(missing)
        condition['retrieval_by_depth'] = {}
        for depth in sorted({r['question']['depth'] for r in rows}):
            selected = [r for r in rows if r['question']['depth'] == depth]
            condition['retrieval_by_depth'][str(depth)] = {
                'assigned': len(selected),
                'found': sum(r.get('retrieval', {}).get('found', 0) for r in selected),
                'required': sum(len(r['question']['required_fact_ids']) for r in selected),
                'complete': sum(r.get('retrieval', {}).get('complete', False) for r in selected)}
        conditions[name] = condition
    differences = {}
    for suffix in ('RAG', 'ORACLE'):
        left = [r for r in all_rows if r['condition'] == 'R-'+suffix]
        right = [r for r in all_rows if r['condition'] == 'G-'+suffix]
        if left and right:
            differences[suffix] = paired_bootstrap(left, right)
            if suffix == 'RAG':
                for depth in (0, 1, 2, 3):
                    differences[f'RAG-depth-{depth}'] = paired_bootstrap(
                        [r for r in left if r['question']['depth'] == depth],
                        [r for r in right if r['question']['depth'] == depth])
    return {'schema': 'editguard-depth-results-v1', 'phase': receipts[0]['plan']['phase'],
            'execution_status': 'completed' if all(r['status'] == 'completed' for r in receipts) else 'incomplete',
            'assigned_records': len(all_rows), 'independent_lineages': len({r['question']['lineage_id'] for r in all_rows}),
            'conditions': conditions, 'paired_differences': differences, 'receipt_audit': checks,
            'model_measurements': {r['plan']['model']: [s.get('runtime') for s in r['sessions'] if s.get('runtime')] for r in receipts},
            'audit_model_calls': 0, 'human_reviewed': False,
            'notes': ['Separate stress test; do not pool with operations-v1 accuracy',
                      'Same bounded iterative frontier retriever and model budgets for both checkpoints',
                      'Oracle subset hash-selected before inference; no targeted failure recovery',
                      'Abstentions/errors/unexecuted remain in every assigned primary denominator',
                      'Complete-support subgroup is descriptive, not a replacement primary denominator']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipts', nargs='+', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous depth results')
    result = audit_and_aggregate(args.receipts, protocol=args.protocol)
    write_receipt(args.output, result)
    print(json.dumps({'phase': result['phase'], 'execution_status': result['execution_status'], 'assigned': result['assigned_records']}))


if __name__ == '__main__':
    main()
