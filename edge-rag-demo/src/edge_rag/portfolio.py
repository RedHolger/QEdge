"""Installed, offline portfolio demo, fixture timing, and regression evaluation."""
import argparse
from datetime import datetime, timezone
from importlib.resources import files
import hashlib
import json
import platform
import statistics
import time

from .assistant import LocalAssistant
from .integration_eval import run_full_integration_evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['demo', 'benchmark', 'evaluate'])
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args(argv)
    if not 1 <= args.trials <= 1000:
        parser.error('--trials must be between 1 and 1000')
    data = files('edge_rag').joinpath('release_data')
    catalog = data.joinpath('catalog.jsonl')
    cases = data.joinpath('decision_cases_80.json')
    query = 'dining chair with oak material under 60cm wide'
    if args.command == 'evaluate':
        result = run_full_integration_evaluation(str(catalog), str(cases))
        result['verdict'] = 'PASS' if (
            result['ten_scenarios_evaluation']['all_passed']
            and result['tenant_isolation_evaluation']['isolation_verified']
            and result['decision_80_cases_evaluation']['splits']['overall']['accuracy'] == 1.0
        ) else 'FAIL'
        result['release_scope'] = 'authored internal regression; no independent generalization claim'
        print(json.dumps(result, indent=2))
        return 0 if result['verdict'] == 'PASS' else 1
    assistant = LocalAssistant(catalog_path=str(catalog))
    if args.command == 'demo':
        result = assistant.find(query)
        print(assistant.format_output(result))
        return 0 if result.action == 'RECOMMEND' else 1
    samples = []
    for _ in range(args.trials):
        start = time.perf_counter()
        result = assistant.find(query)
        samples.append((time.perf_counter() - start) * 1000)
        if result.action != 'RECOMMEND':
            raise RuntimeError('fixture decision failed during benchmark')
    print(json.dumps({
        'kind': 'fixture_assistant_timing', 'measured': True,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'hardware': platform.platform(), 'processor': platform.machine(),
        'model': None, 'precision': None, 'generation_tokens': 0,
        'query': query, 'query_characters': len(query), 'trials': args.trials,
        'catalog_sha256': hashlib.sha256(catalog.read_bytes()).hexdigest(),
        'median_ms': statistics.median(samples), 'samples_ms': samples,
        'scope': 'reused assistant, including first request; includes retrieval, verification and rendering; excludes construction and process startup',
        'npu_status': 'UNVERIFIED', 'retrieval_quality': 'NOT MEASURED',
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
