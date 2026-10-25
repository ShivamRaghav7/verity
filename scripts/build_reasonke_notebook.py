"""Build a Colab notebook pinned to a committed EditGuard source revision."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        ast.parse(source)
        result.update(execution_count=None, outputs=[])
    return result


def build(commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Supply the full committed source SHA")
    cells = [cell("markdown", """
        # EditGuard — Released ReasonKE model

        This notebook loads the author's pinned `YukinoKaren/ReasonKE_plus` checkpoint
        for **text questions using editable approved facts**. No training is launched.
        The model and algorithm are upstream contributions. Our integration adds the
        website, versioned approvals, comparison, conflict handling and withdrawal.

        Select **Runtime → Change runtime type → T4 GPU**, then run cells in order.
        This transfers approximately **15.23 GB of source weight tensors**, even
        though inference uses NF4 weights with FP16 computation. Actual runtime,
        peak memory and generated answers are measured here; no T4 fit or accuracy
        is established by this notebook existing. Keep no other model process on
        the GPU. The five predefined development questions include two-hop reasoning;
        all supporting facts are supplied. This is not a held-out accuracy benchmark.

        The author's checkpoint has no model card explicitly tying it to the paper's
        95.48% result. That published number is never shown as our measured score.
        Runtime files are temporary. There is no public tunnel or Drive mount.
    """), cell("code", f"""
        from pathlib import Path
        import json, os, secrets, shutil, subprocess, sys, time, urllib.request
        from datetime import datetime, timezone
        import torch

        SOURCE_COMMIT = {commit!r}
        REPOSITORY_URL = 'https://github.com/ShivamRaghav7/verity.git'
        REPO = Path('/content/editguard-reasonke')
        if not torch.cuda.is_available():
            raise RuntimeError('Select a GPU runtime before downloading the model.')
        gpu = torch.cuda.get_device_properties(0)
        inventory = {{'started_utc': datetime.now(timezone.utc).isoformat(),
            'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': gpu.name, 'gpu_bytes': gpu.total_memory,
            'free_disk_bytes': shutil.disk_usage('/content').free,
            'source_commit': SOURCE_COMMIT}}
        print(json.dumps(inventory, indent=2))
        if inventory['free_disk_bytes'] < 24 * 1024**3:
            raise RuntimeError('At least 24 GiB of free runtime disk is required for this download candidate.')
    """), cell("code", """
        if REPO.exists():
            actual = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
            if actual != SOURCE_COMMIT:
                raise RuntimeError('Existing checkout differs; preserve it and use a fresh runtime.')
        else:
            subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', REPOSITORY_URL, str(REPO)], check=True)
            subprocess.run(['git', '-C', str(REPO), 'checkout', '--detach', SOURCE_COMMIT], check=True)
        os.chdir(REPO)
        VENV = Path('/content/editguard-reasonke-venv')
        if not VENV.exists():
            subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', '--without-pip', str(VENV)], check=True)
        PYTHON = str(VENV / 'bin/python')
        subprocess.run([PYTHON, '-m', 'pip', 'install', '--disable-pip-version-check',
                        '--index-url', 'https://pypi.org/simple', '-r', 'requirements-reasonke.txt'], check=True)
        subprocess.run([PYTHON, '-m', 'unittest', 'tests.test_reasonke_runtime', 'tests.test_chat_reasonke', '-q'], check=True)
        print('Contract tests passed; they do not establish model answer accuracy.')
    """), cell("code", """
        # Explicitly launch ONE GPU process: load, five finite checks, then serve.
        # Re-running this cell preserves the existing process instead of loading twice.
        RUN = REPO / 'artifacts/reasonke/colab-development-1'
        RUN.mkdir(parents=True, exist_ok=True)
        (RUN / 'inventory.json').write_text(json.dumps(inventory, indent=2))
        if 'WORKER' in globals() and WORKER.poll() is None:
            print('The existing worker is still running; use the next progress cell.')
        else:
            TOKEN = secrets.token_urlsafe(32)
            ENV = dict(os.environ, EDITGUARD_REASONKE_TOKEN=TOKEN,
                EDITGUARD_REASONKE_URL='http://127.0.0.1:8010',
                HF_HUB_DOWNLOAD_TIMEOUT='180', TOKENIZERS_PARALLELISM='false')
            LOG = RUN / 'worker.log'
            log_handle = LOG.open('a', encoding='utf-8')
            WORKER = subprocess.Popen([PYTHON, '-u', '-m', 'editguard_reasonke',
                '--manifest', 'configs/reasonke-model.json',
                '--profile', 'configs/reasonke-acceptance.json',
                '--output', str(RUN / 'acceptance.json'), '--resume', '--serve', '--port', '8010'],
                cwd=REPO, env=ENV, stdout=log_handle, stderr=subprocess.STDOUT)
            log_handle.close()
            WORKER_STARTED = time.monotonic()
            print('Worker launched. The next cell shows real progress without another model load.')
    """), cell("code", """
        # Run again to inspect progress. No indefinite waiting or hidden retry loop.
        print(LOG.read_text(encoding='utf-8', errors='replace')[-10000:])
        if WORKER.poll() is not None:
            raise RuntimeError('Worker exited with code ' + str(WORKER.returncode) + '; inspect the log above.')
        try:
            request = urllib.request.Request('http://127.0.0.1:8010/status',
                headers={'Authorization': 'Bearer ' + TOKEN})
            with urllib.request.urlopen(request, timeout=5) as response:
                STATUS = json.load(response)
            print(json.dumps(STATUS, indent=2))
        except OSError:
            if time.monotonic() - WORKER_STARTED > 1800:
                WORKER.terminate()
                raise RuntimeError('The 30-minute deployment-check cap was reached. Logs and completed cases were preserved.')
            print('Model download/load or finite checks are still running. Re-run this progress cell later.')
        receipt = RUN / 'acceptance.json'
        if receipt.exists():
            report = json.loads(receipt.read_text())
            print(json.dumps(report, indent=2)[-18000:])
    """), cell("markdown", """
        ## Open the website after the worker is ready

        Read the actual five generated answers and any failures above. The website
        can be inspected even when some development answers are incorrect; it is
        not labelled scientifically validated by successful loading alone.

        **Chat** accepts normal text questions. **Knowledge** publishes approved
        corrections with scope/source. Ask the question again, use **Compare**, and
        inspect supplied context. Withdraw a correction to remove it from future
        requests; history remains visible. These controls change external memory,
        not model weights. Generated prose and explanatory steps are unverified.
    """), cell("code", """
        request = urllib.request.Request('http://127.0.0.1:8010/status',
            headers={'Authorization': 'Bearer ' + TOKEN})
        with urllib.request.urlopen(request, timeout=5) as response:
            ready = json.load(response)
        if not ready.get('ready'):
            raise RuntimeError('The actual GPU worker is not ready.')
        from google.colab import output
        from urllib.parse import urlsplit
        proxy_url = output.eval_js('google.colab.kernel.proxyPort(8005)')
        proxy = urlsplit(proxy_url)
        origin = proxy.scheme + '://' + proxy.netloc
        if 'WEBSITE' not in globals() or WEBSITE.poll() is not None:
            site_log = (RUN / 'website.log').open('a', encoding='utf-8')
            WEBSITE = subprocess.Popen([PYTHON, '-u', '-m', 'editguard_chat.server',
                '--backend', 'reasonke', '--port', '8005', '--storage', str(RUN / 'chat'),
                '--trusted-origin', origin], cwd=REPO, env=ENV,
                stdout=site_log, stderr=subprocess.STDOUT)
            site_log.close()
        output.serve_kernel_port_as_iframe(8005, height=1000)
    """), cell("code", """
        # Optional private evidence export. Does not include model weights or tokens.
        import zipfile
        bundle = Path('/content/editguard-reasonke-development-evidence.zip')
        with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in ('inventory.json', 'acceptance.json', 'worker.log', 'website.log'):
                path = RUN / name
                if path.exists():
                    archive.write(path, name)
            archive.write(REPO / 'configs/reasonke-model.json', 'model-manifest.json')
            archive.write(REPO / 'configs/reasonke-acceptance.json', 'predefined-cases.json')
        from google.colab import files
        files.download(str(bundle))
    """)]
    return {"cells": cells, "metadata": {"accelerator": "GPU",
        "colab": {"name": "04_reasonke_chat.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args(argv)
    path = ROOT / 'notebooks/04_reasonke_chat.ipynb'
    path.write_text(json.dumps(build(args.commit), indent=1) + '\n', encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
