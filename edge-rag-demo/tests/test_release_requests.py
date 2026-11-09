import unittest
from pathlib import Path
from edge_rag.assistant import LocalAssistant

class ReleaseRequestTests(unittest.TestCase):
    def test_malformed_and_resource_limited_requests(self):
        assistant = LocalAssistant(catalog_path=Path(__file__).parent / 'fixtures/catalog.jsonl')
        for query in ['', ' ', None, {}, 'x' * 4097]:
            for mode in ['find', 'check']:
                with self.subTest(query=repr(query)[:20], mode=mode):
                    result = assistant.run(mode, query, item_id='fixture-chair-oak')
                    self.assertEqual(result.status, 'ERROR')
                    self.assertEqual(result.error['code'], 'INVALID_INPUT')
        for kwargs in [dict(k=101),dict(k=True),dict(top_verify=-1),dict(timeout_s=float('nan')),dict(timeout_s=-1)]:
            with self.subTest(kwargs=kwargs):
                result = assistant.find('chair', **kwargs)
                self.assertEqual(result.error['code'], 'INVALID_INPUT')
