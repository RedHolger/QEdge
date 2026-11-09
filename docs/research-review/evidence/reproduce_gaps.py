"""Read-only source audit probes; writes only stdout and in-memory SQLite."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'edge-rag-demo/src'), str(ROOT / 'qedge-common/src')]
from edge_rag.cli import run_pipeline
from edge_rag.verifier import Verifier
from edge_rag.store import EvidenceStore
from qedge_common.schema import validate
fixture = ROOT / 'edge-rag-demo/tests/fixtures/catalog.jsonl'
v = Verifier(catalog_path=fixture)
results = {'parser': {q: v.parse_requirements(q) for q in ['red chair', 'chair under 60', 'not oak chair', 'chair under 60cm']}}
r = run_pipeline('red chair', catalog_path=fixture)
results['red_chair_verifications'] = r['verified_results']
s = EvidenceStore()
s.upload('private-doc', 'alice', ['private original'])
results['bob_reads_alice'] = s.retrieve('private-doc', 'bob')
s.upload('private-doc', 'bob', ['unauthorized replacement'], expected_version=1)
results['bob_overwrites_alice'] = s.retrieve('private-doc', 'alice')
base = ROOT / 'qedge-common/tests/fixtures/valid'
d = json.loads((base / 'decision.json').read_text())
d.update(status='SUPPORTED', action='RECOMMEND', claims=[])
validate('decision', d)
results['empty_supported_decision_schema_accepted'] = True
h = json.loads((base / 'hardware_status.json').read_text())
h['qnn']['device_verified'] = False
h['npu_status'] = 'measured'
validate('hardware_status', h)
results['unverified_measured_hardware_schema_accepted'] = True
print(json.dumps(results, indent=2))
