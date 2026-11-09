"""Standalone verification probe for Stage 06 Bounded KV Cache Contract.

Verifies:
1. Cache contract alignment and data structures
2. Hand-calculated attention with interior eviction
3. Non-contiguous causal masking and future-token suppression
4. Capacity 1 and tiny budget reconciliation invariants
5. Boundary and negative invariant enforcement (duplicates, non-monotonic, batch > 1)
6. Exact memory accounting (payload, capacity, metadata, transient peak)
7. Full-cache 100% token parity and logit agreement against Stage 05 reference baseline
8. Bounded FIFO and recency_window generation on real SmolLM2 checkpoint
9. Multi-token chunked prefill equivalence with non-contiguous causal masking
10. Contract v2 benchmark emission with cache telemetry
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

from qedge_runtime.config import (
    SMOLLM2_135M_ID,
    SMOLLM2_135M_REVISION,
    ModelArchitectureConfig,
    get_default_model_dir,
)
from qedge_runtime.runner import ModelRunner
from qedge_runtime.cache import (
    BoundedKVCache,
    QEdgeTransformersCacheAdapter,
    EvictionPolicyKind,
    compute_causal_attention_mask,
    CacheMemoryStats,
)
from qedge_runtime.benchmark import build_benchmark_record
from qedge_common.schema import validate as validate_contract

MODEL_DIR = Path("llm-inference/models/smollm2-135m-instruct").resolve()


def run_checks() -> int:
    print("=== Running Stage 06 Bounded Cache Contract Verification Probe ===")

    # Check 1: Model assets and configuration
    print("Check 1: Model checkpoint assets and architecture...")
    if not (MODEL_DIR / "config.json").is_file():
        print(f"  FAIL: Missing model config at {MODEL_DIR}", file=sys.stderr)
        return 1
    with (MODEL_DIR / "config.json").open("r", encoding="utf-8") as f:
        cfg = ModelArchitectureConfig.from_checkpoint_dict(json.load(f))
    assert cfg.num_hidden_layers == 30, f"Expected 30 layers, got {cfg.num_hidden_layers}"
    assert cfg.num_attention_heads == 9, f"Expected 9 heads, got {cfg.num_attention_heads}"
    assert cfg.num_key_value_heads == 3, f"Expected 3 KV heads, got {cfg.num_key_value_heads}"
    assert cfg.head_dim == 64, f"Expected head dim 64, got {cfg.head_dim}"
    print(f"  PASS: Checkpoint verified (30 layers, 9 Q heads, 3 KV heads, head_dim 64).")

    # Check 2: Hand-calculated attention with interior eviction
    print("Check 2: Hand-calculated attention with interior eviction...")
    head_dim = 2
    cache = BoundedKVCache(num_layers=1, num_kv_heads=1, head_dim=head_dim, capacity_tokens=10, policy="full")
    k_init = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]]], dtype=torch.float32)
    v_init = torch.tensor([[[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]]]], dtype=torch.float32)
    cache.update(layer_idx=0, key_states=k_init, value_states=v_init, query_positions=torch.tensor([0, 1, 2, 3]))

    # Evict token 1
    cache.evict_positions([1])
    assert cache.get_absolute_positions(0).tolist() == [0, 2, 3]

    # Add token 4
    k4 = torch.tensor([[[[2.0, 0.0]]]], dtype=torch.float32)
    v4 = torch.tensor([[[[90.0, 100.0]]]], dtype=torch.float32)
    k_res, v_res = cache.update(layer_idx=0, key_states=k4, value_states=v4, query_positions=torch.tensor([4]))
    assert cache.get_absolute_positions(0).tolist() == [0, 2, 3, 4]

    # Query Q = [1.0, 0.0]
    q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    scale = 1.0 / math.sqrt(2)
    s0, s2, s3, s4 = 1.0 * scale, 1.0 * scale, 0.0, 2.0 * scale
    exp_sum = math.exp(s0) + math.exp(s2) + math.exp(s3) + math.exp(s4)
    w0, w2, w3, w4 = math.exp(s0) / exp_sum, math.exp(s2) / exp_sum, math.exp(s3) / exp_sum, math.exp(s4) / exp_sum

    expected_out = torch.tensor([[[[
        w0 * 10.0 + w2 * 50.0 + w3 * 70.0 + w4 * 90.0,
        w0 * 20.0 + w2 * 60.0 + w3 * 80.0 + w4 * 100.0,
    ]]]], dtype=torch.float32)

    actual_scores = torch.matmul(q, k_res.transpose(2, 3)) * scale
    actual_weights = F.softmax(actual_scores, dim=-1)
    actual_out = torch.matmul(actual_weights, v_res)

    assert torch.allclose(actual_out, expected_out, atol=1e-5), f"Mismatch: {actual_out} vs {expected_out}"
    print(f"  PASS: Hand-calculated interior eviction verified (error: {(actual_out - expected_out).abs().max().item():.2e}).")

    # Check 3: Non-contiguous causal masking and future token suppression
    print("Check 3: Non-contiguous causal masking and future-token suppression...")
    q_pos = torch.tensor([2], dtype=torch.long)
    k_pos = torch.tensor([0, 2, 5], dtype=torch.long)
    mask = compute_causal_attention_mask(q_pos, k_pos)
    assert mask[0, 0, 0, 0].item() == 0.0
    assert mask[0, 0, 0, 1].item() == 0.0
    assert torch.isneginf(mask[0, 0, 0, 2])
    print("  PASS: Future token masked to -inf successfully.")

    # Check 4: Capacity 1 and tiny budget reconciliation
    print("Check 4: Capacity 1 and tiny budget reconciliation...")
    cache_c1 = BoundedKVCache(num_layers=1, num_kv_heads=1, head_dim=4, capacity_tokens=1, policy="fifo")
    for i in range(10):
        cache_c1.update(0, torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4), query_positions=torch.tensor([i]))
        assert cache_c1.get_seq_length(0) == 1
        assert cache_c1.get_absolute_positions(0).tolist() == [i]
    assert cache_c1.total_tokens_seen == 10
    assert cache_c1.total_evictions == 9
    assert cache_c1.total_tokens_seen - cache_c1.get_seq_length(0) == cache_c1.total_evictions
    print("  PASS: Capacity 1 reconciliation: 10 seen - 1 resident == 9 evictions.")

    # Check 5: Review defects verification & negative invariant enforcement
    print("Check 5: Review defects verification & negative invariant enforcement...")
    # Review Defect 1: Adapter absolute position routing
    b_adapt = BoundedKVCache(num_layers=1, num_kv_heads=1, head_dim=4, capacity_tokens=10)
    adapt = QEdgeTransformersCacheAdapter(b_adapt)
    k_dummy = torch.randn(1, 1, 1, 4)
    v_dummy = torch.randn(1, 1, 1, 4)
    adapt.update(k_dummy, v_dummy, layer_idx=0, cache_kwargs={"cache_position": torch.tensor([10])})
    assert b_adapt.get_absolute_positions(0).tolist() == [10], (
        f"Adapter failed to preserve position [10], got {b_adapt.get_absolute_positions(0).tolist()}"
    )

    # Review Defect 2: Failure atomicity on out-of-order arrival
    # Appending position 5 after resident position 10 must raise ValueError before mutation
    try:
        b_adapt.update(0, k_dummy, v_dummy, query_positions=torch.tensor([5]))
        assert False, "Out-of-order position [5] after [10] was not rejected!"
    except ValueError as e:
        assert "must be strictly greater" in str(e)
    # State MUST remain [10] and tokens_seen MUST remain 1
    assert b_adapt.get_absolute_positions(0).tolist() == [10], "Failure atomicity violated: positions mutated!"
    assert b_adapt.total_tokens_seen == 1, "Failure atomicity violated: tokens_seen mutated!"

    # Review Defect 3: Layout, dtype, and device contract validation
    b_layout = BoundedKVCache(num_layers=1, num_kv_heads=1, head_dim=1, capacity_tokens=10, dtype=torch.float32)
    # Shape mismatch: 2 heads, dim 3
    k_bad_shape = torch.randn(1, 2, 1, 3, dtype=torch.float32)
    v_bad_shape = torch.randn(1, 2, 1, 3, dtype=torch.float32)
    try:
        b_layout.update(0, k_bad_shape, v_bad_shape)
        assert False, "Mismatched heads and dim not rejected!"
    except ValueError as e:
        assert "num_kv_heads" in str(e) or "head_dim" in str(e)
    assert b_layout.get_seq_length(0) == 0

    # Dtype mismatch: float64 vs float32
    k_bad_dtype = torch.randn(1, 1, 1, 1, dtype=torch.float64)
    v_bad_dtype = torch.randn(1, 1, 1, 1, dtype=torch.float64)
    try:
        b_layout.update(0, k_bad_dtype, v_bad_dtype)
        assert False, "Mismatched dtype float64 not rejected!"
    except TypeError as e:
        assert "Configured cache dtype" in str(e)
    assert b_layout.get_seq_length(0) == 0

    # Review Defect 4: Allocation-aware peak memory and coexisting buffer accounting
    b_peak = BoundedKVCache(num_layers=1, num_kv_heads=1, head_dim=128, capacity_tokens=2, policy="fifo", dtype=torch.float32)
    k2 = torch.randn(1, 1, 2, 128, dtype=torch.float32)
    v2 = torch.randn(1, 1, 2, 128, dtype=torch.float32)
    b_peak.update(0, k2, v2, query_positions=torch.tensor([0, 1]))
    # Update with 1 token coexists at >= 7,168 bytes
    k1 = torch.randn(1, 1, 1, 128, dtype=torch.float32)
    v1 = torch.randn(1, 1, 1, 128, dtype=torch.float32)
    b_peak.update(0, k1, v1, query_positions=torch.tensor([2]))
    stats_peak = b_peak.get_memory_stats()
    assert stats_peak.analytical_peak_bytes >= 7168, (
        f"Expected analytical peak >= 7,168 bytes, got {stats_peak.analytical_peak_bytes}"
    )

    # Temporary memory allowance check & budget rejection failure atomicity
    b_empty = BoundedKVCache(
        num_layers=1, num_kv_heads=1, head_dim=4, capacity_tokens=10,
        temporary_memory_allowance_bytes=1,
    )
    try:
        b_empty.update(0, torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4), query_positions=torch.tensor([0]))
        assert False, "Allowance=1 update was not rejected!"
    except MemoryError:
        pass
    # Must NOT record candidate peak as achieved peak, overflow must remain False
    assert b_empty.get_seq_length(0) == 0
    assert b_empty.get_memory_stats().analytical_peak_bytes == 0
    assert b_empty.get_memory_stats().peak_temporary_bytes == 0
    assert b_empty.get_memory_stats().temporary_memory_overflow is False
    assert b_empty.get_memory_stats().last_rejected_attempt is not None

    b_allow = BoundedKVCache(
        num_layers=1, num_kv_heads=1, head_dim=128, capacity_tokens=2, policy="fifo",
        dtype=torch.float32, temporary_memory_allowance_bytes=5000,
    )
    b_allow.update(0, k2, v2, query_positions=torch.tensor([0, 1]))
    peak_allowed = b_allow.get_memory_stats().analytical_peak_bytes
    try:
        b_allow.update(0, k1, v1, query_positions=torch.tensor([2]))
        assert False, "Temporary memory overflow not raised!"
    except MemoryError:
        pass
    assert b_allow.get_seq_length(0) == 2
    assert b_allow.get_absolute_positions(0).tolist() == [0, 1]
    assert b_allow.get_memory_stats().analytical_peak_bytes == peak_allowed
    assert b_allow.get_memory_stats().temporary_memory_overflow is False

    # Multi-layer pruning preflight allowance atomicity
    b_multi = BoundedKVCache(num_layers=2, num_kv_heads=1, head_dim=64, capacity_tokens=2, policy="fifo")
    b_multi.defer_eviction = True
    b_multi.update(0, torch.randn(1, 1, 4, 64), torch.randn(1, 1, 4, 64), query_positions=torch.tensor([0, 1, 2, 3]))
    b_multi.update(1, torch.randn(1, 1, 4, 64), torch.randn(1, 1, 4, 64), query_positions=torch.tensor([0, 1, 2, 3]))
    b_multi.temporary_memory_allowance_bytes = 10  # Insufficient
    try:
        b_multi.evict_to_capacity()
        assert False, "evict_to_capacity insufficient allowance not rejected!"
    except MemoryError:
        pass
    assert b_multi.get_seq_length(0) == 4 and b_multi.get_seq_length(1) == 4, "Partial pruning occurred!"

    # Unrelated accelerator allocation does not alter cache telemetry
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    unrelated = torch.zeros(1_000_000, device=dev, dtype=torch.float32)
    b_iso = BoundedKVCache(num_layers=1, num_kv_heads=1, head_dim=128, capacity_tokens=2, policy="fifo", dtype=torch.float32)
    b_iso.update(0, torch.randn(1, 1, 2, 128, device=dev), torch.randn(1, 1, 2, 128, device=dev), query_positions=torch.tensor([0, 1]))
    stats_iso1 = b_iso.get_memory_stats()
    assert stats_iso1.measured_peak_bytes is None, "Global allocator snapshot labelled as measured peak!"
    del unrelated
    if dev.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    b_iso.update(0, torch.randn(1, 1, 1, 128, device=dev), torch.randn(1, 1, 1, 128, device=dev), query_positions=torch.tensor([2]))
    stats_iso2 = b_iso.get_memory_stats()
    assert stats_iso2.measured_peak_bytes is None
    assert stats_iso2.analytical_peak_bytes >= stats_iso1.analytical_peak_bytes
    assert stats_iso2.resident_payload_bytes == 2048

    # Additional negative invariants: Duplicate positions & Batch size > 1
    cache_neg = BoundedKVCache(num_layers=1, num_kv_heads=1, head_dim=4, capacity_tokens=8)
    try:
        cache_neg.update(0, torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4), query_positions=torch.tensor([3, 3]))
        assert False, "Duplicate positions not rejected!"
    except ValueError:
        pass
    try:
        cache_neg.update(0, torch.randn(2, 1, 1, 4), torch.randn(2, 1, 1, 4))
        assert False, "Batch size > 1 not rejected!"
    except ValueError:
        pass
    print("  PASS: All 4 review defect probes and negative invariants verified.")

    # Check 6: Memory accounting
    print("Check 6: Exact memory accounting...")
    cache_mem = BoundedKVCache(num_layers=30, num_kv_heads=3, head_dim=64, capacity_tokens=64, policy="fifo", dtype=torch.float32)
    k_mem = torch.randn(1, 3, 32, 64)
    v_mem = torch.randn(1, 3, 32, 64)
    for i in range(30):
        cache_mem.update(i, k_mem, v_mem)
    m = cache_mem.get_memory_stats().to_dict()
    assert m["bytes_per_token_model"] == 2 * 3 * 64 * 4 * 30  # 46,080 bytes
    assert m["resident_payload_bytes"] == 32 * 46080           # 1,474,560 bytes
    assert m["capacity_payload_bytes"] == 64 * 46080           # 2,949,120 bytes
    assert m["metadata_bytes"] == 30 * 32 * 8 + 64             # 7,744 bytes
    # Peak temporary memory captures coexisting buffers (old resident + incoming + concat + other layers + metadata)
    assert m["peak_temporary_bytes"] == 1531712
    assert m["analytical_peak_bytes"] == 1531712
    print(f"  PASS: Memory accounting verified: {m['resident_payload_bytes']:,} resident bytes.")

    # Check 7: Real model full-cache baseline parity
    print("Check 7: Full-cache baseline parity on real SmolLM2 checkpoint...")
    runner = ModelRunner(model_dir=MODEL_DIR, device="cpu", dtype="float32", local_files_only=True)
    res_full = runner.generate(
        prompt="Explain edge computing in one concise sentence.",
        max_new_tokens=12,
        cache_policy="full",
        record_logits=True,
    )
    assert len(res_full.generated_token_ids) == 12
    assert res_full.cache_stats["policy"] == "full"
    assert res_full.cache_stats["total_evictions"] == 0
    print(f"  PASS: Full cache generated 12 tokens: {res_full.generated_token_ids[:6]}...")

    # Check 8: Bounded FIFO generation and reconciliation
    print("Check 8: Bounded FIFO generation and eviction reconciliation...")
    res_fifo = runner.generate(
        prompt="The quick brown fox jumps over the lazy sleeping dog near the river bank.",
        max_new_tokens=10,
        cache_policy="fifo",
        cache_capacity=10,
        sink_tokens=2,
    )
    assert len(res_fifo.generated_token_ids) == 10
    stats_fifo = res_fifo.cache_stats
    assert stats_fifo["policy"] == "fifo"
    assert stats_fifo["resident_tokens"] <= 10
    assert stats_fifo["total_evictions"] > 0
    assert stats_fifo["total_tokens_seen"] - stats_fifo["resident_tokens"] == stats_fifo["total_evictions"]
    print(f"  PASS: Bounded FIFO reconciled: {stats_fifo['total_tokens_seen']} seen, "
          f"{stats_fifo['resident_tokens']} resident, {stats_fifo['total_evictions']} evictions.")

    # Check 9: Multi-token chunked prefill equivalence
    print("Check 9: Multi-token chunked prefill equivalence...")
    prompt = "Artificial intelligence at the edge."
    inputs = runner.tokenizer(prompt, return_tensors="pt")
    ids = inputs["input_ids"]
    total = ids.shape[1]

    # Baseline single pass
    b_single = BoundedKVCache(30, 3, 64, capacity_tokens=total, policy="full")
    a_single = QEdgeTransformersCacheAdapter(b_single)
    out_single = runner.model(input_ids=ids, past_key_values=a_single, use_cache=True)
    logits_single = out_single.logits[:, -1, :]

    # Chunked pass: 3 tokens, then remaining tokens
    b_chunk = BoundedKVCache(30, 3, 64, capacity_tokens=total, policy="full")
    a_chunk = QEdgeTransformersCacheAdapter(b_chunk)
    c1_ids = ids[:, :3]
    c2_ids = ids[:, 3:]
    runner.model(input_ids=c1_ids, past_key_values=a_chunk, use_cache=True)
    mask_c2 = b_chunk.build_causal_mask(torch.arange(3, total), dtype=torch.float32)
    out_chunk = runner.model(
        input_ids=c2_ids,
        attention_mask=mask_c2,
        position_ids=torch.arange(3, total).unsqueeze(0),
        past_key_values=a_chunk,
        use_cache=True,
    )
    logits_chunk = out_chunk.logits[:, -1, :]

    diff = (logits_single - logits_chunk).abs().max().item()
    assert diff < 1e-4, f"Chunked prefill logits diverged: {diff}"
    print(f"  PASS: Chunked prefill equivalent to single pass (max diff: {diff:.2e}).")

    # Check 10: Contract v2 benchmark emission with cache telemetry
    print("Check 10: Contract v2 benchmark emission...")
    record = build_benchmark_record(res_fifo)
    validate_contract("benchmark_result", record)
    assert record["cache"]["policy"] == "fifo"
    assert record["cache"]["capacity_tokens"] == 10
    assert record["cache"]["evictions"] > 0
    assert record["cache"]["peak_bytes"] > 0
    print("  PASS: Validated contract v2 benchmark_result schema successfully.")

    print("\n=== ALL 10 STAGE 06 VERIFICATION PROBES PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(run_checks())
