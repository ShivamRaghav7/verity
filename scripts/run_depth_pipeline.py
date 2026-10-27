"""Run one finite depth-study phase on the two checkpoints sequentially."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('profile', 'development', 'final'), required=True)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    if args.phase == 'final' and not args.protocol:
        parser.error('Final depth inference requires a frozen protocol')
    args.directory.mkdir(parents=True, exist_ok=True)
    for model in ('reasonke', 'instruction'):
        command = [sys.executable, '-u', '-m', 'editguard_depth.run', '--model', model,
                   '--phase', args.phase, '--output', str(args.directory/f'{args.phase}-{model}.json'), '--resume']
        if args.protocol:
            command += ['--protocol', str(args.protocol)]
        print('Starting', args.phase, model, flush=True)
        with (args.directory/f'{args.phase}-{model}.log').open('a', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=31000)
        if result.returncode:
            raise SystemExit('Depth phase stopped. Preserve receipts; no second model started.')
    print('Both depth jobs completed; factual outcomes remain recorded.', flush=True)


if __name__ == '__main__':
    main()
