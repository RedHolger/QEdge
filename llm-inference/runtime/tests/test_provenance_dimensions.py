"""Dimension consistency at artifact reuse and verified loading boundaries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import onnx
from onnx import helper, TensorProto
from qedge_runtime.config import compute_file_sha256
from qedge_runtime.qualcomm.provenance import (
    validate_onnx_export_provenance, ProvenanceValidationError,
    EXPECTED_MODEL_ID, EXPECTED_SOURCE_REVISION,
)
from qedge_runtime.qualcomm.export import export_smollm2_to_onnx
from qedge_runtime.qualcomm.runner import ONNXModelRunner


class DimensionContractTests(unittest.TestCase):
    def test_exact_dimensions_and_loading_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'smollm2-135m-instruct.onnx'
            graph = helper.make_graph(
                [helper.make_node('Constant', [], ['logits'], value=helper.make_tensor(
                    'value', TensorProto.FLOAT, [1, 1, 49152], [0.] * 49152))],
                'dimension_probe',
                [helper.make_tensor_value_info('input_ids', TensorProto.INT64, ['batch', 'sequence'])],
                [helper.make_tensor_value_info('logits', TensorProto.FLOAT, [1, 1, 49152])])
            onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=10), artifact)
            manifest = dict(model_id=EXPECTED_MODEL_ID, source_revision=EXPECTED_SOURCE_REVISION,
                            export_format='ONNX', opset_version=17, onnx_file=artifact.name,
                            sha256=compute_file_sha256(artifact), file_size_bytes=artifact.stat().st_size,
                            input_schema={'input_ids': {'dtype': 'int64', 'shape': ['batch', 'sequence']}},
                            output_schema={'logits': {'dtype': 'float32', 'shape': [1, 1, 49152]}})
            path = root / 'export_manifest.json'
            path.write_text(json.dumps(manifest))
            validate_onnx_export_provenance(artifact)
            for section, name, axis, value in [
                ('input_schema', 'input_ids', 0, 99),
                ('input_schema', 'input_ids', 1, -7),
                ('input_schema', 'input_ids', 0, True),
                ('input_schema', 'input_ids', 0, {'bad': 'dimension'}),
                ('input_schema', 'input_ids', 0, ''),
                ('input_schema', 'input_ids', 0, 'bad symbol'),
                ('input_schema', 'input_ids', 0, 'different_symbol'),
                ('output_schema', 'logits', 0, 2),
                ('output_schema', 'logits', 0, 'batch'),
            ]:
                with self.subTest(value=value, section=section):
                    invalid = copy.deepcopy(manifest)
                    invalid[section][name]['shape'][axis] = value
                    path.write_text(json.dumps(invalid))
                    with self.assertRaises(ProvenanceValidationError):
                        validate_onnx_export_provenance(artifact)
                    # Verified loading must fail before tokenizer/session initialization.
                    with patch('qedge_runtime.qualcomm.runner.check_model_assets', return_value=(True, [])):
                        with self.assertRaises(ValueError):
                            ONNXModelRunner(artifact, model_dir=root)
                    # Export must reject reuse and reach the rebuild boundary.
                    with patch('qedge_runtime.qualcomm.export.check_model_assets', return_value=(True, [])), \
                         patch('qedge_runtime.qualcomm.export.AutoModelForCausalLM.from_pretrained', side_effect=RuntimeError('rebuild required')):
                        with self.assertRaisesRegex(RuntimeError, 'rebuild required'):
                            export_smollm2_to_onnx(model_dir=root, output_dir=root)

    def test_unknown_graph_dimension_rejected(self):
        from qedge_runtime.qualcomm.provenance import validate_dimensions
        dimension = onnx.TensorShapeProto.Dimension()
        for value in [None, 'batch', 1]:
            with self.assertRaises(ProvenanceValidationError):
                validate_dimensions([value], [dimension], 'unknown')
