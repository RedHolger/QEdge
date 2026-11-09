"""Relocate package sources, build wheels, install and test offline in isolation.

Run bootstrap_portfolio.sh first to acquire build/dependency caches explicitly.
Only disposable directories are created; repository environments are untouched.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
logs = root / 'handoff/research-review/12-release-review/logs'
logs.mkdir(parents=True, exist_ok=True)
env = {k: v for k, v in os.environ.items() if not k.startswith(('QEDGE_', 'PYTHONPATH', 'VIRTUAL_ENV'))}
with tempfile.TemporaryDirectory(prefix='qedge-release-') as temporary:
    work = Path(temporary)
    for package in ['qedge-common', 'edge-rag-demo']:
        target = work / package
        target.mkdir()
        for name in ['pyproject.toml', 'README.md']:
            shutil.copy2(root / package / name, target / name)
        for name in ['src', 'tests', 'schemas']:
            source = root / package / name
            if source.exists():
                shutil.copytree(source, target / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        subprocess.run(['uv','build','--offline','--wheel','--out-dir',str(work/'wheels'),str(target)],check=True,env=env,cwd=work)
    subprocess.run(['uv','venv','--python','3.11',str(work/'env')],check=True,env=env,cwd=work)
    python = work / 'env/bin/python'
    subprocess.run(['uv','pip','install','--offline','--python',str(python),*[str(p) for p in (work/'wheels').glob('*.whl')]],check=True,env=env,cwd=work)
    # Remove relocated sources before execution to expose undeclared source imports.
    shutil.rmtree(work/'edge-rag-demo')
    shutil.rmtree(work/'qedge-common')
    code = '''
import socket, sys
from edge_rag.portfolio import main
from edge_rag import __file__ as source
assert 'site-packages' in source, source
def deny(*args, **kwargs):
    raise RuntimeError('Network forbidden during offline release verification')
socket.socket.connect = deny
socket.create_connection = deny
raise SystemExit(main(sys.argv[1:]))
'''
    for command in ['demo','benchmark','evaluate']:
        with (logs / f'offline-installed-{command}.log').open('w') as output:
            subprocess.run([str(python),'-I','-c',code,command],check=True,stdout=output,stderr=subprocess.STDOUT,env=env,cwd=work)
    subprocess.run(['uv','pip','check','--python',str(python)],check=True,env=env,cwd=work)
    print(json.dumps({'relocated_source_build':'PASS','source_removed_before_execution':True,'isolated_python':True,'network_denied_at_socket_connect':True,'installed_commands':['demo','benchmark','evaluate']},indent=2))
