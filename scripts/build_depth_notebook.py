"""Build the staged, pinned Colab depth-challenge launcher without running it."""
import argparse
import json
from pathlib import Path
import re
from scripts.build_reasonke_notebook import cell


def build(commit):
    if not re.fullmatch('[a-f0-9]{40}', commit):
        raise ValueError('Supply a committed full source SHA')
    cells = [cell('markdown', '''
        # EditGuard: separate one-, two- and three-hop depth challenge

        This is a new study, not a replacement score for the completed operations-v1
        experiment. All operational assignments are fictional Indian-context facts.
        Thirty-two new held-out lineages contain 256 questions, including 64 three-hop
        questions. Question templates are partitioned by development/final split.

        First run the hardware/setup cells, then the **16-request profile only**.
        Model loading uses pinned upstream weights, approximately 15.23 GB of source
        tensors per checkpoint; no training runs. Do not run another GPU worker in
        this runtime. Colab GPU quota is an external requirement, not something this
        notebook bypasses. All subsequent phases have explicit disabled switches.
    '''), cell('code', f'''
        from pathlib import Path
        import json, os, shutil, subprocess, sys, time
        import torch, psutil
        SOURCE_COMMIT = {commit!r}
        REPO = Path('/content/editguard-depth')
        if not torch.cuda.is_available():
            raise RuntimeError('A GPU-enabled runtime is required. Resolve Colab availability before proceeding.')
        print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
        owned = [p.pid for p in psutil.process_iter(['cmdline']) if any(x in (p.info['cmdline'] or [])
            for x in ('editguard_depth.run', 'editguard_benchmark.run', 'editguard_reasonke'))]
        if owned: raise RuntimeError('Use an idle runtime; existing model/evaluation processes: ' + str(owned))
        if shutil.disk_usage('/content').free < 24*1024**3: raise RuntimeError('At least 24 GiB free disk required.')
        print('CUDA:', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0), '| source:', SOURCE_COMMIT)
    '''), cell('code', '''
        if REPO.exists():
            actual = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
            if actual != SOURCE_COMMIT: raise RuntimeError('Preserve the different checkout; use a fresh runtime.')
        else:
            subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/ShivamRaghav7/verity.git',str(REPO)],check=True)
            subprocess.run(['git','-C',str(REPO),'checkout','--detach',SOURCE_COMMIT],check=True)
        os.chdir(REPO)
        VENV = Path('/content/editguard-depth-venv')
        subprocess.run([sys.executable,'-m','venv','--system-site-packages','--without-pip',str(VENV)],check=True)
        PYTHON = str(VENV/'bin/python')
        subprocess.run([PYTHON,'-m','pip','install','--disable-pip-version-check','-r','requirements-reasonke.txt'],check=True)
        subprocess.run([PYTHON,'-m','unittest','tests.test_depth_data','tests.test_depth_retrieval','tests.test_depth_runner','-q'],check=True)
        DIRECTORY = REPO/'artifacts/depth-run'
        DIRECTORY.mkdir(parents=True,exist_ok=True)
        print('Software/graph checks passed; they are not neural accuracy measurements.')
    '''), cell('code', '''
        # First question: how much time/memory do actual 1/2/3-hop generations use?
        # Fixed inputs: development only; outputs: two 8-request raw ledgers+logs.
        if 'DEPTH_JOB' in globals() and DEPTH_JOB.poll() is None:
            print('Existing depth phase still running; no duplicate launched.')
        else:
            handle = (DIRECTORY/'pipeline.log').open('a',encoding='utf-8')
            DEPTH_JOB = subprocess.Popen([PYTHON,'-u','-m','scripts.run_depth_pipeline','--phase','profile','--directory',str(DIRECTORY)],cwd=REPO,stdout=handle,stderr=subprocess.STDOUT)
            handle.close()
            print('Profile only: 8 ReasonKE then 8 Qwen requests. No training; no final answers accessed.')
    '''), cell('code', '''
        import math
        print('Current phase exit:',DEPTH_JOB.poll(),'(None means running)')
        print((DIRECTORY/'pipeline.log').read_text(errors='replace')[-1000:])
        for phase in ('profile','development','final'):
            for model in ('reasonke','instruction'):
                path=DIRECTORY/(phase+'-'+model+'.json')
                if path.exists():
                    r=json.loads(path.read_text())
                    print(phase,model,r['status'],sum(x['outcome']!='unexecuted' for x in r['records']),'/',len(r['records']))
                    if phase == 'profile':
                        measured=[x for x in r['records'] if x.get('response')]
                        timings=[x.get('pipeline_seconds') for x in measured]
                        if (r['status']=='completed' and len(measured)==8 and
                            all(isinstance(t,(int,float)) and math.isfinite(t) and t>0 for t in timings)):
                            mean=sum(timings)/8
                            generation_peaks=[x['response'].get('generation',{}).get('peak_allocated_bytes') for x in measured]
                            generation_peaks=[x for x in generation_peaks if isinstance(x,int)]
                            loads=[{'load_wall_seconds':s.get('load_wall_seconds'),
                                'runtime_load_seconds':s.get('runtime',{}).get('load_seconds'),
                                'hardware':s.get('runtime',{}).get('hardware')}
                                for s in r.get('sessions',[]) if s.get('runtime')]
                            print(json.dumps({'profile_budget':model,'measured_requests':8,
                                'pipeline_sum_seconds':sum(timings),'pipeline_mean_seconds':mean,
                                'pipeline_max_seconds':max(timings),'final_requests':272,
                                'forecast_272_mean_seconds':272*mean,'forecast_272_max_seconds':272*max(timings),
                                'forecast_excludes':'Model loading, downloads and later variation; only eight development samples',
                                'load_sessions':loads,'generation_peak_allocated_bytes':max(generation_peaks,default=None)},indent=2))
                        else:
                            print('No complete eight-response profile budget yet:',model)
    '''), cell('markdown', '''
        ## Review actual profiles before the next phase

        Profile: 8 requests/checkpoint; development: 24/checkpoint; final: 272/checkpoint
        (256 matched questions + 16 preselected three-hop oracle cases). A new finite
        runtime estimate must use the actual depth profiles, not the old two-hop
        study. Model loading/downloading is separate and can vary. Hard limits are
        1,800 seconds loading, 240 seconds/request and 8 hours/model; these are watchdog
        caps, not estimates. Each outcome is saved atomically. Exact-plan resume
        skips every completed correct/wrong/abstained/error record.
    '''), cell('code', '''
        RUN_DEVELOPMENT = False
        if not RUN_DEVELOPMENT:
            print('Development disabled until both profiles finish and their budget is reviewed.')
        elif DEPTH_JOB.poll() is None or DEPTH_JOB.returncode:
            raise RuntimeError('Complete the profile phase first.')
        else:
            handle=(DIRECTORY/'development-pipeline.log').open('a',encoding='utf-8')
            DEPTH_JOB=subprocess.Popen([PYTHON,'-u','-m','scripts.run_depth_pipeline','--phase','development','--directory',str(DIRECTORY)],cwd=REPO,stdout=handle,stderr=subprocess.STDOUT)
            handle.close()
    '''), cell('code', '''
        RUN_FREEZE = False
        PROTOCOL = DIRECTORY/'protocol.json'
        if not RUN_FREEZE:
            print('Freeze disabled until both development ledgers complete and settings are final.')
        elif DEPTH_JOB.poll() is None or DEPTH_JOB.returncode:
            raise RuntimeError('Complete development first.')
        else:
            subprocess.run([PYTHON,'-m','editguard_depth.run','--freeze-directory',str(DIRECTORY),'--output',str(PROTOCOL)],cwd=REPO,check=True)
    '''), cell('code', '''
        RUN_FINAL = False
        if not RUN_FINAL:
            print('Final disabled. Review the frozen profile-grounded estimate before enabling.')
        elif not PROTOCOL.exists():
            raise RuntimeError('Identical frozen protocol+profile/development inputs required.')
        elif DEPTH_JOB.poll() is None:
            raise RuntimeError('A phase is already running.')
        else:
            handle=(DIRECTORY/'final-pipeline.log').open('a',encoding='utf-8')
            DEPTH_JOB=subprocess.Popen([PYTHON,'-u','-m','scripts.run_depth_pipeline','--phase','final','--protocol',str(PROTOCOL),'--directory',str(DIRECTORY)],cwd=REPO,stdout=handle,stderr=subprocess.STDOUT)
            handle.close()
            print('544 assigned final generations; no outcome-based retries or test tuning.')
    '''), cell('code', '''
        RUN_SCORE = False
        RESULTS = DIRECTORY/'results.json'
        if not RUN_SCORE:
            print('Scoring disabled until both final ledgers finish. All assigned outcomes remain in the denominator.')
        elif DEPTH_JOB.poll() is None or DEPTH_JOB.returncode:
            raise RuntimeError('Complete the final phase successfully before scoring.')
        elif RESULTS.exists():
            print('Existing audited scorecard preserved:',RESULTS)
        else:
            subprocess.run([PYTHON,'-m','editguard_depth.metrics',
                str(DIRECTORY/'final-reasonke.json'),str(DIRECTORY/'final-instruction.json'),
                '--protocol',str(PROTOCOL),'--output',str(RESULTS)],cwd=REPO,check=True)
    '''), cell('code', '''
        # Export bounded raw evidence after each phase; no weights, tokens or chats.
        import zipfile,base64
        from IPython.display import display,HTML
        bundle=Path('/content/editguard-depth-evidence.zip')
        with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(DIRECTORY.glob('*.json')): archive.write(path,path.name)
        print('Phase exit:',DEPTH_JOB.poll(),'| archive bytes:',bundle.stat().st_size)
        display(HTML('<a download="editguard-depth-evidence.zip" href="data:application/zip;base64,'+base64.b64encode(bundle.read_bytes()).decode()+'">Download depth evidence</a>'))
    ''')]
    return {'cells': cells,'metadata':{'accelerator':'GPU','colab':{'name':'06_depth_challenge.ipynb','provenance':[]},
            'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True)
    args=parser.parse_args()
    target=Path(__file__).resolve().parents[1]/'notebooks/06_depth_challenge.ipynb'
    target.write_bytes((json.dumps(build(args.commit),indent=1)+'\n').encode())
    print(target)


if __name__=='__main__': main()
