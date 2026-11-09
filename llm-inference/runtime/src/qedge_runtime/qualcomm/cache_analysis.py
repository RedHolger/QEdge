"""Analysis of KV cache dimensions, processor transfers, and serialized graph constraints (Stage 11 - Repaired).

Provides:
- Derived KV cache payload byte counts strictly based on verified SmolLM2-135M configuration.
- Clear distinction between one-way tensor payload and round-trip bus traffic.
- Reporting of physical device measurements as UNAVAILABLE / NOT MEASURED (hardware absent).
- Explicit analytical bus transfer modeling exposing all formulas and stated assumptions.
- Accurately scoped architectural conclusions limited to the implemented graph and runtime,
  avoiding universal assertions about all serialized implementations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class CachePayloadBreakdown:
    """Exact byte counts derived from verified model architecture dimensions."""

    layers: int
    kv_heads: int
    head_dim: head_dim = 64
    resident_tokens: int = 512
    bytes_per_fp16: int = 2
    bytes_per_int8: int = 1

    # Derived byte counts
    fp16_one_way_payload_bytes: int = 0
    fp16_one_way_payload_mib: float = 0.0
    fp16_round_trip_traffic_bytes: int = 0
    fp16_round_trip_traffic_mib: float = 0.0

    int8_one_way_payload_bytes: int = 0
    int8_one_way_payload_mib: float = 0.0
    int8_round_trip_traffic_bytes: int = 0
    int8_round_trip_traffic_mib: float = 0.0

    derivation_formula: str = ""


@dataclass
class AnalyticalTransferEstimate:
    """Theoretical transfer latency estimate under explicit, stated assumptions."""

    assumed_bus_bandwidth_gb_s: float
    assumed_driver_overhead_ms: float
    round_trip_traffic_bytes: int
    estimated_transfer_ms: float
    formula: str
    is_measured: bool = False
    notes: str = ""


@dataclass
class CacheSurgeryAnalysis:
    """Breakdown of cache surgery constraints in serialized graphs and host-accelerator buses."""

    implemented_graph_cache_mode: str
    cache_selection_supported_in_implemented_graph: bool
    selection_execution_processor: str
    kv_tensor_crosses_processors_if_host_managed: bool
    payload_breakdown: CachePayloadBreakdown
    measured_device_transfer_ms: Optional[float]
    measured_npu_decode_compute_ms: Optional[float]
    measured_transfer_to_compute_ratio: Optional[float]
    analytical_transfer_estimates: list[AnalyticalTransferEstimate]
    architecture_findings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_kv_cache_payload(
    layers: int = 30,
    kv_heads: int = 3,
    head_dim: int = 64,
    resident_tokens: int = 512,
) -> CachePayloadBreakdown:
    """Calculate exact dense KV cache payload bytes and round-trip traffic."""
    # SmolLM2-135M config: 30 layers, 3 KV heads, 64 head dimension
    # Dense K+V one-way payload = 2 * layers * kv_heads * resident_tokens * head_dim * bytes_per_element
    fp16_one_way = 2 * layers * kv_heads * resident_tokens * head_dim * 2
    fp16_round_trip = 2 * fp16_one_way

    int8_one_way = 2 * layers * kv_heads * resident_tokens * head_dim * 1
    int8_round_trip = 2 * int8_one_way

    formula = (
        "one_way_payload_bytes = 2 * layers * kv_heads * resident_tokens * head_dim * bytes_per_elem; "
        "round_trip_traffic_bytes = 2 * one_way_payload_bytes (device-to-host + host-to-device)"
    )

    return CachePayloadBreakdown(
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        resident_tokens=resident_tokens,
        bytes_per_fp16=2,
        bytes_per_int8=1,
        fp16_one_way_payload_bytes=fp16_one_way,
        fp16_one_way_payload_mib=round(fp16_one_way / (1024 * 1024), 3),
        fp16_round_trip_traffic_bytes=fp16_round_trip,
        fp16_round_trip_traffic_mib=round(fp16_round_trip / (1024 * 1024), 3),
        int8_one_way_payload_bytes=int8_one_way,
        int8_one_way_payload_mib=round(int8_one_way / (1024 * 1024), 3),
        int8_round_trip_traffic_bytes=int8_round_trip,
        int8_round_trip_traffic_mib=round(int8_round_trip / (1024 * 1024), 3),
        derivation_formula=formula,
    )


def analyze_cache_surgery_in_serialized_graphs() -> CacheSurgeryAnalysis:
    """Analyze memory, bus traffic, and execution placement for KV cache eviction."""
    payload = calculate_kv_cache_payload(layers=30, kv_heads=3, head_dim=64, resident_tokens=512)

    # Expose analytical transfer estimates across stated bus bandwidth assumptions (FP16 round-trip: 22.5 MiB)
    estimates = []
    round_trip_bytes = payload.fp16_round_trip_traffic_bytes
    for bw_gb_s, overhead_ms in [(2.0, 0.5), (4.0, 0.5), (8.0, 0.3)]:
        raw_xfer_ms = (round_trip_bytes / (bw_gb_s * 1e9)) * 1000.0
        total_ms = raw_xfer_ms + overhead_ms
        estimates.append(
            AnalyticalTransferEstimate(
                assumed_bus_bandwidth_gb_s=bw_gb_s,
                assumed_driver_overhead_ms=overhead_ms,
                round_trip_traffic_bytes=round_trip_bytes,
                estimated_transfer_ms=round(total_ms, 2),
                formula="estimated_transfer_ms = (round_trip_traffic_bytes / (bandwidth_gb_s * 1e9) * 1000) + overhead_ms",
                is_measured=False,
                notes=f"Theoretical analytical estimate assuming effective {bw_gb_s} GB/s bus bandwidth; NOT a hardware measurement.",
            )
        )

    findings = [
        (
            "Implemented graph status: The exported ONNX model operates in 'use_cache=False' mode, "
            "recomputing the entire prompt/prefix on every step. It does not export or maintain "
            "in-graph KV cache tensors, meaning in-graph cache selection is not currently supported."
        ),
        (
            "Execution placement: In an architecture where cache pruning is managed by the host CPU "
            "(e.g. Python or C++ runtime) while decode compute executes on a discrete accelerator, "
            "cache selection algorithms operate on the host processor outside the accelerator graph."
        ),
        (
            "Bus traffic vs. payload: For SmolLM2-135M at 512 tokens FP16, the one-way dense KV cache "
            f"is {payload.fp16_one_way_payload_mib} MiB ({payload.fp16_one_way_payload_bytes:,} bytes). "
            f"If the host reads the cache to prune tokens and writes it back, round-trip bus traffic is "
            f"{payload.fp16_round_trip_traffic_mib} MiB ({payload.fp16_round_trip_traffic_bytes:,} bytes). "
            "For INT8, one-way payload is 5.625 MiB and round-trip traffic is 11.25 MiB."
        ),
        (
            "Measurement status: Physical transfer latency and accelerator decode latency cannot be "
            "measured in the current environment because Qualcomm Snapdragon hardware is absent. "
            "All physical latency values are recorded as UNAVAILABLE."
        ),
        (
            "Architectural scope: Host-side cache eviction incurs bus transfer overhead on systems with "
            "discrete interconnects (e.g. FastRPC or PCIe). However, this transfer overhead can be "
            "avoided by alternative deployment patterns, such as stateful in-graph KV cache buffers, "
            "native in-graph accelerator eviction operators, hardware-managed paging, or unified memory "
            "architectures where host CPU and accelerator share physical DRAM without bus copies."
        ),
    ]

    return CacheSurgeryAnalysis(
        implemented_graph_cache_mode="full_prefix_recomputation_no_kv_cache",
        cache_selection_supported_in_implemented_graph=False,
        selection_execution_processor="Host CPU (if externally managed)",
        kv_tensor_crosses_processors_if_host_managed=True,
        payload_breakdown=payload,
        measured_device_transfer_ms=None,
        measured_npu_decode_compute_ms=None,
        measured_transfer_to_compute_ratio=None,
        analytical_transfer_estimates=estimates,
        architecture_findings=findings,
    )
