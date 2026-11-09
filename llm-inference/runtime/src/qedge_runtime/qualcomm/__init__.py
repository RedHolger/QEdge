"""Qualcomm deployment and execution validation package for QEdge.

Provides:
- Hardware, software, driver, and licensed artifact inventory.
- Primary-documentation compatibility matrix and single deployment route selection.
- Independently executable ONNX model export and CPU parity verification.
- Analysis of cache selection constraints and memory transfers in serialized graphs.
- Executable Qualcomm device deployment runbook and governance.
"""
from __future__ import annotations

from .compatibility import get_qualcomm_compatibility_matrix
from .inventory import (
    QualcommHardwareInventory,
    inventory_qualcomm_hardware_and_software,
)
from .cache_analysis import (
    CacheSurgeryAnalysis,
    analyze_cache_surgery_in_serialized_graphs,
)
from .export import (
    ONNXExportResult,
    export_smollm2_to_onnx,
    verify_onnx_cpu_numerical_parity,
)
from .runner import ONNXModelRunner

__all__ = [
    "QualcommHardwareInventory",
    "inventory_qualcomm_hardware_and_software",
    "get_qualcomm_compatibility_matrix",
    "CacheSurgeryAnalysis",
    "analyze_cache_surgery_in_serialized_graphs",
    "ONNXExportResult",
    "export_smollm2_to_onnx",
    "verify_onnx_cpu_numerical_parity",
    "ONNXModelRunner",
]
