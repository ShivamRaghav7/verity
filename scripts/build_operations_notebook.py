"""Build the source-pinned sequential-model operations evaluation notebook."""
import argparse
import json
from pathlib import Path

from scripts.build_reasonke_notebook import build as base_notebook, cell


def build(commit):
    base = base_notebook(commit)
    base["cells"] = [cell("markdown", """
        # EditGuard — controlled operations benchmark

        This notebook profiles and evaluates the implemented five-condition
        comparison. No training runs. The author ReasonKE checkpoint and official
        Qwen instruction checkpoint are loaded sequentially on one T4, never together.
        Keep a separate runtime from the chat demo so no other model occupies VRAM.

        The dataset contains 48 isolated edit lineages split before wording
        augmentation. Final inference requires a source/data/profile-bound frozen
        protocol. All wrong, abstained, failed and unexecuted outcomes are retained.
        A wrong answer is not retried. Runtime storage is temporary: export receipts
        after each stage. No published accuracy number is substituted for our run.
    """), *base["cells"][1:3], cell("code", """
        # Finite profile: 12 actual requests per checkpoint, sequentially.
        # Both source tensors are about 15.23 GB; a fresh second checkpoint needs disk.
        DIRECTORY = REPO / 'artifacts/operations'
        DIRECTORY.mkdir(parents=True, exist_ok=True)
        PROFILE_LOG = DIRECTORY / 'profile-pipeline.log'
        if 'PROFILE_JOB' in globals() and PROFILE_JOB.poll() is None:
            print('The existing profile pipeline is running. Do not launch another.')
        else:
            handle = PROFILE_LOG.open('a', encoding='utf-8')
            PROFILE_JOB = subprocess.Popen([PYTHON, '-u', '-m', 'scripts.run_operations_pipeline',
                '--phase', 'profile', '--directory', str(DIRECTORY)], cwd=REPO,
                env=dict(os.environ, HF_HUB_DOWNLOAD_TIMEOUT='180', TOKENIZERS_PARALLELISM='false'),
                stdout=handle, stderr=subprocess.STDOUT)
            handle.close()
            print('One profile pipeline launched; 24 total assigned model generations.')
    """), cell("code", """
        print(PROFILE_LOG.read_text(errors='replace')[-3000:])
        for name in ('reasonke', 'instruction'):
            path = DIRECTORY / ('profile-' + name + '.json')
            if path.exists():
                receipt = json.loads(path.read_text())
                done = [row for row in receipt['records'] if row['outcome'] != 'unexecuted']
                print(name, 'status:', receipt['status'], 'recorded:', len(done), 'assigned:', len(receipt['records']))
                if done: print('Latest:', done[-1]['id'], done[-1]['outcome'], done[-1].get('pipeline_seconds'))
            else: print(name, 'not started yet')
        if PROFILE_JOB.poll() is not None:
            print('Pipeline exit:', PROFILE_JOB.returncode)
    """), cell("markdown", """
        ## Development and freeze

        Inspect profiles and the measured resource budget first. Run development
        through the same pipeline with phase `development` (120 assigned requests).
        Use development only for prompt/retrieval/scoring choices. If source or
        settings change, profiles and protocol identities must be refreshed under
        a declared revised run; old receipts stay preserved.

        Freeze only after settings are fixed. The protocol records actual profile
        means/maxima and extrapolations for 784 ReasonKE and 528 instruction final
        requests, including 16 oracle-support questions per model. Those forecasts
        are uncertain and exclude cold downloads/loading. Review them before the
        substantial final command. No final test tuning or answer-based retries.
    """), cell("code", """
        # This cell only seals the fixed protocol; it performs zero generations.
        PROTOCOL = DIRECTORY / 'protocol.json'
        if PROTOCOL.exists():
            print('Existing protocol preserved:', PROTOCOL)
        else:
            completed = subprocess.run([PYTHON, '-m', 'editguard_benchmark.run', '--freeze-profiles',
                str(DIRECTORY/'profile-reasonke.json'), str(DIRECTORY/'profile-instruction.json'),
                '--output', str(PROTOCOL)], cwd=REPO, text=True, capture_output=True)
            print(completed.stdout, completed.stderr)
            completed.check_returncode()
    """), cell("code", """
        # Explicit finite final stage. Enable after reviewing the profile-grounded budget.
        RUN_FINAL = False
        if not RUN_FINAL:
            print('Final generation is disabled until the measured budget is reviewed.')
        elif 'FINAL_JOB' in globals() and FINAL_JOB.poll() is None:
            print('The existing final pipeline is running.')
        else:
            handle = (DIRECTORY/'final-pipeline.log').open('a', encoding='utf-8')
            FINAL_JOB = subprocess.Popen([PYTHON, '-u', '-m', 'scripts.run_operations_pipeline',
                '--phase', 'final', '--protocol', str(PROTOCOL), '--directory', str(DIRECTORY)],
                cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
            handle.close()
            print('Final pipeline launched; exactly 1,312 assigned requests, no training.')
    """), cell("code", """
        # Run after the two final jobs have ended; all outcomes remain in denominators.
        RESULTS = REPO/'reports/reasonke/operations-final/results.json'
        completed = subprocess.run([PYTHON, '-m', 'editguard_benchmark.metrics',
            str(DIRECTORY/'final-reasonke.json'), str(DIRECTORY/'final-instruction.json'),
            '--output', str(RESULTS)], cwd=REPO, text=True, capture_output=True)
        print(completed.stdout, completed.stderr)
        completed.check_returncode()
    """), cell("code", """
        # Download bounded raw receipts and inputs. No weights, chat database or tokens.
        import zipfile
        bundle = Path('/content/editguard-operations-evidence.zip')
        with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(DIRECTORY.glob('*.json')):
                archive.write(path, str(path.relative_to(REPO)))
            for relative in ('configs/operations-evaluation.json', 'configs/instruction-model.json',
                             'configs/reasonke-model.json', 'data/reasonke/operations-v1/manifest.json'):
                archive.write(REPO/relative, relative)
            results = REPO/'reports/reasonke/operations-final/results.json'
            if results.exists(): archive.write(results, str(results.relative_to(REPO)))
        from google.colab import files
        files.download(str(bundle))
    """)]
    base["metadata"]["colab"]["name"] = "05_operations_benchmark.ipynb"
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "notebooks/05_operations_benchmark.ipynb"
    path.write_text(json.dumps(build(args.commit), indent=1) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
