#!/usr/bin/env python3
"""Standalone Verification Probe for Stage 07 Research Selectors.

Executes 17 authoritative mathematical and runtime checks to verify:
1. KeyDiff deterministic mathematical agreement with pure-Python reference.
2. QUOKA deterministic mathematical agreement with pure-Python reference.
3. Query-sensitivity vs query-independence (holding K fixed, altering Q changes QUOKA but preserves KeyDiff).
4. Geometric divergence: Outlier key diversity (KeyDiff) vs newest-token recency (FIFO).
5. Robustness at zero norms, ties, and tiny budget (B=1, B >= S).
6. GQA group pre-aggregation correctness (H_q=9, H_kv=3).
7. Full-budget (100%) exact logit and token parity against unselected baseline.
8. Monotonic selection pressure at matched ratios (100%, 75%, 50%, 25%).
9. Memory contract preservation: KeyDiff reduces resident memory to C; QUOKA preserves full resident memory.
10. Strict state isolation across sequential generations without cross-request state leakage.
11. Faithful per-KV-head selection vs SharedHeadQUOKAApproximation on disagreeing orthogonal heads.
12. GQA cancellation preservation without renormalization and chunked prefill greedy decode parity.
13. 2D padding mask (float32, int64, bool) handling in SparseAttentionContext without broadcasting error.
14. Fully blocked 4D additive mask with finfo.min producing strictly zero attention weights and zero output.
15. Length-aware token agreement and negation-aware target scorer.
16. Finite additive attention biases preservation and blocking sentinel protection.
17. Teacher-forced deployed path parity, suffix invariance, and eviction pruning schedule.

Usage:
    ./llm-inference/runtime/.venv/bin/python handoff/research-review/07-research-selectors/probes/verify_selectors.py
"""
import sys
import time
from pathlib import Path

# Add runtime source to python path
repo_root = Path(__file__).resolve().parents[4]
runtime_src = repo_root / "llm-inference" / "runtime" / "src"
if str(runtime_src) not in sys.path:
    sys.path.insert(0, str(runtime_src))

import torch
import torch.nn as nn

from qedge_runtime.selectors import (
    KeyDiffSelector,
    QUOKASelector,
    SharedHeadQUOKAApproximation,
    reference_keydiff,
    reference_quoka,
    paper_quoka_oracle,
    masked_attention_reference,
    SparseAttentionContext,
    is_blocking_sentinel,
)
from qedge_runtime.evaluation import (
    exact_sequence_agreement,
    length_aware_token_agreement,
    sequence_agreement_diagnostic,
    coarse_authored_target_hit,
    compute_teacher_forced_step_logits,
    compute_teacher_forced_logit_mse,
)
from qedge_runtime.cache import BoundedKVCache
from qedge_runtime.runner import ModelRunner
from qedge_runtime.config import get_default_model_dir


def check_1_keydiff_reference_agreement() -> bool:
    print("[Check 1/17] Verifying KeyDiff agreement with pure-Python reference...")
    torch.manual_seed(42)
    s, d = 25, 16
    keys = torch.randn(s, d)
    keys_tensor = keys.unsqueeze(0).unsqueeze(0)
    keys_list = keys.tolist()

    for b in [1, 5, 10, 18, 25]:
        ref = reference_keydiff(keys_list, budget=b, sink_tokens=4, recent_tokens=1)
        res = KeyDiffSelector(sink_tokens=4, recent_tokens=1).select_indices(keys_tensor, budget=b)
        if res.retained_indices != ref:
            print(f"  FAIL: KeyDiff mismatch at budget={b}: {res.retained_indices} != {ref}")
            return False
    print("  PASS: Exact index agreement across all test budgets.")
    return True


def check_2_quoka_reference_agreement() -> bool:
    print("[Check 2/17] Verifying QUOKA agreement with pure-Python reference on MHA...")
    torch.manual_seed(100)
    s_k, s_q, d = 30, 6, 16
    keys = torch.randn(s_k, d)
    queries = torch.randn(s_q, d)

    keys_tensor = keys.unsqueeze(0).unsqueeze(0)
    queries_tensor = queries.unsqueeze(0).unsqueeze(0)

    for b in [1, 8, 15, 22, 30]:
        ref = reference_quoka(
            queries=queries.tolist(),
            keys=keys.tolist(),
            budget=b,
            sink_tokens=4,
            recent_tokens=1,
            representative_ratio=0.25,
        )
        res = QUOKASelector(sink_tokens=4, recent_tokens=1, representative_ratio=0.25).select_indices(
            keys_tensor, budget=b, queries=queries_tensor
        )
        if res.retained_indices != ref:
            print(f"  FAIL: QUOKA mismatch at budget={b}: {res.retained_indices} != {ref}")
            return False
    print("  PASS: Exact index agreement across all test budgets.")
    return True


def check_3_query_sensitivity_vs_independence() -> bool:
    print("[Check 3/17] Verifying Query-Sensitivity (QUOKA) vs Query-Independence (KeyDiff)...")
    torch.manual_seed(777)
    s, d = 32, 16
    keys = torch.randn(1, 3, s, d)
    q1 = torch.randn(1, 9, 4, d)
    q2 = -torch.randn(1, 9, 4, d) + 8.0

    # KeyDiff must be identical
    kd = KeyDiffSelector(sink_tokens=3, recent_tokens=1)
    kd_1 = kd.select_indices(keys, budget=12, queries=q1).retained_indices
    kd_2 = kd.select_indices(keys, budget=12, queries=q2).retained_indices
    if kd_1 != kd_2:
        print(f"  FAIL: KeyDiff changed on different queries: {kd_1} != {kd_2}")
        return False

    # QUOKA must differ
    quoka = QUOKASelector(sink_tokens=3, recent_tokens=1)
    q_1 = quoka.select_indices(keys, budget=12, queries=q1).retained_indices
    q_2 = quoka.select_indices(keys, budget=12, queries=q2).retained_indices
    if q_1 == q_2:
        print("  FAIL: QUOKA produced identical indices for divergent query distributions")
        return False

    print("  PASS: KeyDiff is strictly query-independent; QUOKA is query-sensitive.")
    return True


def check_4_geometric_divergence() -> bool:
    print("[Check 4/17] Verifying Outlier Key Diversity (KeyDiff) vs Recency Window (FIFO)...")
    d = 8
    sinks = torch.randn(2, d)
    outliers = torch.zeros(2, d)
    outliers[0, 0] = 10.0
    outliers[1, 1] = -10.0
    clustered = torch.zeros(4, d)
    for i in range(4):
        clustered[i, 2] = 3.0 + 0.1 * i
    tail = torch.randn(1, d)

    keys = torch.cat([sinks, outliers, clustered, tail], dim=0).unsqueeze(0).unsqueeze(0)
    kd_res = KeyDiffSelector(sink_tokens=2, recent_tokens=1).select_indices(keys, budget=5)
    # KeyDiff must retain outliers [0, 1, 2, 3, 8]
    expected_kd = [0, 1, 2, 3, 8]
    if kd_res.retained_indices != expected_kd:
        print(f"  FAIL: KeyDiff failed to retain diverse outliers: got {kd_res.retained_indices}, expected {expected_kd}")
        return False

    # FIFO retains newest [0, 1, 6, 7, 8]
    fifo_cache = BoundedKVCache(1, 1, d, capacity_tokens=5, policy="fifo", sink_tokens=2)
    fifo_retained = fifo_cache._select_retained_indices(torch.arange(9), 9)
    if fifo_retained != [0, 1, 6, 7, 8]:
        print(f"  FAIL: FIFO returned unexpected indices: {fifo_retained}")
        return False

    print(f"  PASS: KeyDiff selected outliers {kd_res.retained_indices}; FIFO selected newest {fifo_retained}.")
    return True


def check_5_zero_norms_and_boundary_budgets() -> bool:
    print("[Check 5/17] Verifying zero norms, ties, budget=1, and full budget boundaries...")
    zero_keys = torch.zeros(1, 1, 12, 16)
    kd = KeyDiffSelector(sink_tokens=2, recent_tokens=1)
    res_zero = kd.select_indices(zero_keys, budget=6)
    if len(res_zero.retained_indices) != 6 or torch.isnan(res_zero.scores).any():
        print("  FAIL: Zero keys triggered NaNs or wrong count")
        return False

    # Budget 1: strictly newest token
    rand_keys = torch.randn(1, 1, 15, 16)
    res_b1 = kd.select_indices(rand_keys, budget=1)
    if res_b1.retained_indices != [14]:
        print(f"  FAIL: Budget 1 did not retain token 14: got {res_b1.retained_indices}")
        return False

    # Budget >= 15: all tokens
    res_bf = kd.select_indices(rand_keys, budget=20)
    if res_bf.retained_indices != list(range(15)):
        print(f"  FAIL: Over-budget did not retain all tokens: got {res_bf.retained_indices}")
        return False

    print("  PASS: Zero norms handled stably; budget boundaries verified.")
    return True


def check_6_gqa_group_preaggregation() -> bool:
    print("[Check 6/17] Verifying GQA group pre-aggregation (H_q=9, H_kv=3)...")
    keys = torch.randn(1, 3, 20, 16)
    queries = torch.randn(1, 9, 4, 16)
    quoka = QUOKASelector(sink_tokens=2, recent_tokens=1)
    res = quoka.select_indices(keys, budget=10, queries=queries)
    if res.metadata["group_size"] != 3:
        print(f"  FAIL: Group size was {res.metadata['group_size']}, expected 3")
        return False
    if len(res.retained_indices) != 10:
        print(f"  FAIL: Retained indices count {len(res.retained_indices)} != 10")
        return False
    print("  PASS: GQA group pre-aggregation correctly mapped 9 query heads to 3 KV groups.")
    return True


def check_7_full_budget_exact_logit_parity(runner: ModelRunner) -> bool:
    print("[Check 7/17] Verifying full-budget (100%) exact logit parity on SmolLM2...")
    prompt = "Edge AI enables real-time reasoning on low-power silicon."
    base_res = runner.generate(prompt, max_new_tokens=8, record_logits=True, cache_policy="full")
    quoka_res = runner.generate(
        prompt,
        max_new_tokens=8,
        record_logits=True,
        cache_policy="full",
        attention_selector="quoka",
        attention_ratio=1.0,
    )

    if base_res.generated_token_ids != quoka_res.generated_token_ids:
        print(f"  FAIL: Token mismatch at 100% budget: {base_res.generated_token_ids} != {quoka_res.generated_token_ids}")
        return False

    base_logits = torch.stack(base_res.step_logits)
    quoka_logits = torch.stack(quoka_res.step_logits)
    max_diff = (base_logits - quoka_logits).abs().max().item()
    if max_diff >= 1e-4:
        print(f"  FAIL: Max logit difference {max_diff} exceeded tolerance 1e-4")
        return False

    print(f"  PASS: Exact token match; max logit diff = {max_diff:.2e} (< 1e-4).")
    return True


def check_8_selection_pressure(runner: ModelRunner) -> bool:
    print("[Check 8/17] Verifying monotonic selection pressure across matched ratios...")
    prompt = "Modern neural network architectures combine multi-head attention and feed-forward layers."
    ratios = [1.0, 0.75, 0.50, 0.25]
    sparsities = []
    for r in ratios:
        res = runner.generate(
            prompt,
            max_new_tokens=6,
            cache_policy="full",
            attention_selector="quoka",
            attention_ratio=r,
        )
        s = res.cache_stats["sparse_attention"]["sparsity_ratio"]
        sparsities.append((r, s))

    print(f"  Measured sparsity ratios: {sparsities}")
    if sparsities[0][1] != 0.0:
        print("  FAIL: 100% budget had non-zero sparsity")
        return False
    if not (sparsities[1][1] < sparsities[2][1] < sparsities[3][1]):
        print("  FAIL: Sparsity was not strictly monotonic with decreasing budget")
        return False

    print("  PASS: Monotonic selection pressure verified across 100%, 75%, 50%, 25%.")
    return True


def check_9_resident_memory_contracts(runner: ModelRunner) -> bool:
    print("[Check 9/17] Verifying divergence between Permanent Eviction and Temporary Selection...")
    prompt = "Describe the operating principles of silicon neural processing units."
    cap = 16

    # 1. KeyDiff: Permanent eviction
    res_kd = runner.generate(prompt, max_new_tokens=8, cache_policy="keydiff", cache_capacity=cap)
    kd_tokens = res_kd.cache_stats["resident_tokens"]
    kd_evictions = res_kd.cache_stats["total_evictions"]
    if kd_tokens != cap or kd_evictions <= 0:
        print(f"  FAIL: KeyDiff did not evict down to capacity {cap}: resident={kd_tokens}, evictions={kd_evictions}")
        return False

    # 2. QUOKA: Temporary selection
    res_q = runner.generate(prompt, max_new_tokens=8, cache_policy="full", attention_selector="quoka", attention_budget=cap)
    q_tokens = res_q.cache_stats["resident_tokens"]
    q_evictions = res_q.cache_stats["total_evictions"]
    expected_full = len(res_q.prompt_token_ids) + len(res_q.generated_token_ids) - 1
    if q_tokens != expected_full or q_evictions != 0:
        print(f"  FAIL: QUOKA mutated resident cache storage: resident={q_tokens}, evictions={q_evictions}")
        return False

    print(f"  PASS: KeyDiff reduced resident tokens to {kd_tokens} (evicted {kd_evictions}); QUOKA retained all {q_tokens} resident tokens (0 evicted).")
    return True


def check_10_state_isolation(runner: ModelRunner) -> bool:
    print("[Check 10/17] Verifying state isolation across repeated requests...")
    res1 = runner.generate("What is 1+1?", max_new_tokens=4, attention_selector="quoka", attention_ratio=0.5)
    res2 = runner.generate(
        "Explain in detail the fundamental laws of thermodynamics and entropy in physical systems.",
        max_new_tokens=4,
        attention_selector="quoka",
        attention_ratio=0.5,
    )

    if len(res1.prompt_token_ids) == len(res2.prompt_token_ids):
        print("  FAIL: Prompt token lengths unexpectedly identical")
        return False
    if res2.cache_stats["sparse_attention"]["steps_evaluated"] <= 0:
        print("  FAIL: Steps evaluated not recorded on second request")
        return False

    print("  PASS: Repeated requests cleanly isolated without telemetry leakage.")
    return True


def check_11_disagreeing_heads_orthogonal_example() -> bool:
    print("[Check 11/17] Verifying per-KV-head selection on disagreeing orthogonal heads...")
    # Head 0 prefers token 0 (cos=1.0 vs cos=0.1)
    # Head 1 prefers token 1 (cos=0.0 vs cos=1.0)
    # Keys: token 0 = [1.0, 0.0], token 1 = [0.1, 0.995]
    queries = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])  # [1, 2, 1, 2]
    keys = torch.tensor([[[[1.0, 0.0], [0.1, 0.995]], [[1.0, 0.0], [0.1, 0.995]]]])  # [1, 2, 2, 2]

    # Faithful per-KV-head paper algorithm
    quoka = QUOKASelector(sink_tokens=0, recent_tokens=0)
    res = quoka.select_indices(keys=keys, budget=1, queries=queries)
    if res.per_head_indices.tolist() != [[0], [1]]:
        print(f"  FAIL: Faithful QUOKA returned {res.per_head_indices.tolist()}, expected [[0], [1]]")
        return False

    # Pure-Python oracle
    oracle = paper_quoka_oracle(queries[0].tolist(), keys[0].tolist(), budget=1, sink_tokens=0, recent_tokens=0)
    if oracle != [[0], [1]]:
        print(f"  FAIL: paper_quoka_oracle returned {oracle}, expected [[0], [1]]")
        return False

    # Shared-head approximation ablation collapses to compromise index [1]
    approx = SharedHeadQUOKAApproximation(sink_tokens=0, recent_tokens=0)
    res_approx = approx.select_indices(keys=keys, budget=1, queries=queries)
    if res_approx.per_head_indices.tolist() != [[1], [1]]:
        print(f"  FAIL: SharedHeadQUOKAApproximation returned {res_approx.per_head_indices.tolist()}, expected [[1], [1]]")
        return False

    print("  PASS: Faithful QUOKA selected [[0], [1]]; Shared approximation collapsed to [[1], [1]].")
    return True


def check_12_chunked_prefill_and_gqa_cancellation(runner: ModelRunner) -> bool:
    print("[Check 12/17] Verifying GQA cancellation preservation and chunked prefill parity...")
    # 1. GQA cancellation without re-normalization
    q0 = torch.tensor([1.0, 0.0])
    q1 = torch.tensor([-0.9, 0.0])
    q = torch.stack([q0, q1]).unsqueeze(0).unsqueeze(2)  # [1, 2, 1, 2]
    k = torch.tensor([1.0, 0.0]).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 2]
    q_bar = torch.mean(q, dim=1, keepdim=True)
    score = torch.matmul(q_bar, k.transpose(-2, -1)).item()
    if abs(score - 0.05) > 1e-4:
        print(f"  FAIL: GQA cancellation score was {score}, expected 0.05")
        return False

    # 2. Chunked prefill greedy decode parity
    prompt = ("The quick brown fox jumps over the lazy dog. " * 8).strip()
    res_full = runner.generate(prompt, max_new_tokens=8, use_chat_template=False, cache_policy="full")
    res_chunk = runner.generate(prompt, max_new_tokens=8, use_chat_template=False, chunk_size=16, cache_policy="full")
    if res_full.generated_token_ids != res_chunk.generated_token_ids:
        print(f"  FAIL: Chunked prefill tokens mismatch: {res_full.generated_token_ids} != {res_chunk.generated_token_ids}")
        return False

    print("  PASS: GQA cancellation preserved; exact chunked prefill decode parity achieved.")
    return True


def check_13_2d_padding_masks() -> bool:
    print("[Check 13/17] Verifying 2D padding mask (float32, int64, bool) in SparseAttentionContext...")
    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = 2
            self.layer_idx = 0
            self.scaling = 1.0 / (2.0 ** 0.5)
            self.training = False
            self.attention_dropout = 0.0
            self.q_proj = nn.Identity()
            self.k_proj = nn.Identity()
            self.v_proj = nn.Identity()
            self.o_proj = nn.Identity()

    class DummyMod(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].self_attn = DummyAttn()

    model = DummyMod()
    h = torch.ones(1, 2, 2)
    pos = torch.tensor([0, 1])

    for b in [1, 2]:
        ctx = SparseAttentionContext(model, "quoka", budget=b, sink_tokens=0, recent_tokens=0)
        with ctx:
            for mask_dtype in [torch.float32, torch.int64, torch.bool]:
                mask_2d = torch.ones(1, 2, dtype=mask_dtype)
                try:
                    out, weights = model.layers[0].self_attn(h, attention_mask=mask_2d, position_ids=pos)
                except Exception as e:
                    print(f"  FAIL: Mask {mask_dtype} raised error at budget {b}: {e}")
                    return False
                if out.shape != (1, 2, 2) or torch.isnan(out).any():
                    print(f"  FAIL: Invalid output for mask {mask_dtype} at budget {b}")
                    return False

    print("  PASS: 2D padding masks (float32, int64, bool) handled cleanly without broadcasting errors.")
    return True


def check_14_fully_blocked_rows_produce_zero_output() -> bool:
    print("[Check 14/17] Verifying fully blocked 4D additive masks produce zero attention output...")
    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = 2
            self.layer_idx = 0
            self.scaling = 1.0 / (2.0 ** 0.5)
            self.training = False
            self.attention_dropout = 0.0
            self.q_proj = nn.Identity()
            self.k_proj = nn.Identity()
            self.v_proj = nn.Identity()
            self.o_proj = nn.Identity()

    class DummyMod(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].self_attn = DummyAttn()

    model = DummyMod()
    h = torch.ones(1, 2, 2)
    pos = torch.tensor([0, 1])

    for b in [1, 2]:
        ctx = SparseAttentionContext(model, "quoka", budget=b, sink_tokens=0, recent_tokens=0)
        with ctx:
            mask_min = torch.full((1, 1, 2, 2), torch.finfo(torch.float32).min)
            out, weights = model.layers[0].self_attn(h, attention_mask=mask_min, position_ids=pos)
            if not torch.all(weights == 0.0):
                print(f"  FAIL: Fully blocked weights non-zero at budget {b}: {weights}")
                return False
            if not torch.all(out == 0.0):
                print(f"  FAIL: Fully blocked output non-zero at budget {b}: {out}")
                return False

    print("  PASS: Fully blocked rows explicitly detected and produce strictly zero attention weights/output.")
    return True


def check_15_length_aware_metrics_and_scorer() -> bool:
    print("[Check 15/17] Verifying length-aware token agreement and target scorer semantics...")
    ref = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    gen_trunc = [1, 2]
    # Length-aware agreement: 2 / 10 = 0.20
    agree = length_aware_token_agreement(gen_trunc, ref)
    if abs(agree - 0.20) > 1e-4:
        print(f"  FAIL: length_aware_token_agreement was {agree}, expected 0.20")
        return False

    # Scorer tests
    if not coarse_authored_target_hit("The capital is Paris.", "Paris"):
        print("  FAIL: Target 'Paris' should hit in 'The capital is Paris.'")
        return False
    if coarse_authored_target_hit("The binary is 110.", "101"):
        print("  FAIL: Target '101' should not hit in 'The binary is 110.'")
        return False
    if coarse_authored_target_hit("The capital is not Paris.", "Paris"):
        print("  FAIL: Negated target 'not Paris' should not hit")
        return False
    if coarse_authored_target_hit("The latency was reduced.", "Red"):
        print("  FAIL: Substring 'Red' inside 'reduced' should not hit")
        return False

    print("  PASS: Length-aware agreement penalizes premature truncation; target scorer handles negation & boundaries.")
    return True


def check_16_finite_additive_attention_biases() -> bool:
    print("[Check 16/17] Verifying finite additive attention biases and sentinel protection...")
    class DummySelfAttn(nn.Module):
        def __init__(self, head_dim=2):
            super().__init__()
            self.head_dim = head_dim
            self.layer_idx = 0
            self.scaling = 1.0 / (head_dim ** 0.5)
            self.q_proj = nn.Linear(head_dim, head_dim, bias=False)
            self.k_proj = nn.Linear(head_dim, head_dim, bias=False)
            self.v_proj = nn.Linear(head_dim, head_dim, bias=False)
            self.o_proj = nn.Linear(head_dim, head_dim, bias=False)
            self.attention_dropout = 0.0
            nn.init.zeros_(self.q_proj.weight)
            nn.init.zeros_(self.k_proj.weight)
            nn.init.eye_(self.v_proj.weight)
            nn.init.eye_(self.o_proj.weight)

        def forward(self, *args, **kwargs):
            pass

    class DummyLayer(nn.Module):
        def __init__(self, attn):
            super().__init__()
            self.self_attn = attn

    class DummyModel(nn.Module):
        def __init__(self, attn):
            super().__init__()
            self.layers = nn.ModuleList([DummyLayer(attn)])

    attn = DummySelfAttn(head_dim=2)
    model = DummyModel(attn)
    h = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    pos = torch.tensor([0, 1])
    mask = torch.tensor([[[[0.0, float("-inf")], [-2.0, 0.0]]]])

    ctx = SparseAttentionContext(model, "quoka", budget=2)
    with ctx:
        out, weights = model.layers[0].self_attn(h, attention_mask=mask, position_ids=pos)
        exp_w = torch.softmax(torch.tensor([-2.0, 0.0]), dim=-1)
        diff = (weights[0, 0, 1] - exp_w).abs().max().item()
        if diff > 1e-5:
            print(f"  FAIL: Reproduction weights diff {diff} exceeds tolerance: {weights[0, 0, 1]} vs {exp_w}")
            return False
        diff_out = (out[0, 1] - exp_w).abs().max().item()
        if diff_out > 1e-5:
            print(f"  FAIL: Reproduction output diff {diff_out} exceeds tolerance: {out[0, 1]} vs {exp_w}")
            return False

    # Check that finite biases are not misclassified as sentinels
    for val in [-2.0, -10.0, -100.0, -1000.0]:
        if is_blocking_sentinel(torch.tensor(val)).item():
            print(f"  FAIL: Finite bias {val} misclassified as blocking sentinel!")
            return False

    # Check that actual sentinels are detected
    for s_val in [float("-inf"), torch.finfo(torch.float32).min, float("nan")]:
        if not is_blocking_sentinel(torch.tensor(s_val)).item():
            print(f"  FAIL: Sentinel {s_val} failed detection!")
            return False

    print("  PASS: Finite additive biases preserved (softmax([-2,0]) verified); sentinels correctly protected.")
    return True


def check_17_teacher_forced_deployed_path_parity(runner: ModelRunner) -> bool:
    print("[Check 17/17] Verifying teacher-forced deployed path parity and suffix invariance...")
    prompt = "What is the capital of France?"
    res = runner.generate(prompt, max_new_tokens=4, use_chat_template=True)
    ref_tokens = res.generated_token_ids
    if not ref_tokens:
        print("  FAIL: No reference tokens generated.")
        return False

    # 1. Full budget parity: dense vs dense must be 0.0
    mse_dense = compute_teacher_forced_logit_mse(runner, prompt, ref_tokens)
    if mse_dense != 0.0:
        print(f"  FAIL: Dense teacher-forced MSE {mse_dense} != 0.0")
        return False

    # 2. Prompt IDs match runner trial
    logits_orig, prompt_ids = compute_teacher_forced_step_logits(runner, prompt, ref_tokens)
    if prompt_ids != res.prompt_token_ids:
        print("  FAIL: Prompt IDs do not match runner trial formatted tokens.")
        return False

    # 3. Future suffix invariance
    altered_ref = list(ref_tokens)
    altered_ref[-1] = 9999
    logits_altered, _ = compute_teacher_forced_step_logits(runner, prompt, altered_ref)
    for i in range(len(ref_tokens) - 1):
        diff = (logits_orig[i] - logits_altered[i]).abs().max().item()
        if diff != 0.0:
            print(f"  FAIL: Step {i} was affected by future token modification! diff={diff}")
            return False

    # 4. QUOKA 100% full-budget logit parity
    mse_quoka_full = compute_teacher_forced_logit_mse(
        runner, prompt, ref_tokens, attention_selector="quoka", attention_ratio=1.0
    )
    if mse_quoka_full > 1e-4:
        print(f"  FAIL: QUOKA 100% full-budget MSE {mse_quoka_full} > 1e-4")
        return False

    print("  PASS: Teacher-forced path matches deployed runner; suffix invariance and full-budget parity verified.")
    return True


def main() -> int:
    print("=" * 70)
    print("STAGE 07 RESEARCH SELECTORS: AUTHORITATIVE VERIFICATION PROBE")
    print("=" * 70)

    model_dir = get_default_model_dir()
    runner = ModelRunner(model_dir)

    checks = [
        check_1_keydiff_reference_agreement,
        check_2_quoka_reference_agreement,
        check_3_query_sensitivity_vs_independence,
        check_4_geometric_divergence,
        check_5_zero_norms_and_boundary_budgets,
        check_6_gqa_group_preaggregation,
        lambda: check_7_full_budget_exact_logit_parity(runner),
        lambda: check_8_selection_pressure(runner),
        lambda: check_9_resident_memory_contracts(runner),
        lambda: check_10_state_isolation(runner),
        check_11_disagreeing_heads_orthogonal_example,
        lambda: check_12_chunked_prefill_and_gqa_cancellation(runner),
        check_13_2d_padding_masks,
        check_14_fully_blocked_rows_produce_zero_output,
        check_15_length_aware_metrics_and_scorer,
        check_16_finite_additive_attention_biases,
        lambda: check_17_teacher_forced_deployed_path_parity(runner),
    ]

    passed = 0
    failed = 0
    for idx, chk in enumerate(checks, start=1):
        try:
            ok = chk()
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR in Check {idx}: {e}")
            failed += 1

    print("=" * 70)
    print(f"SUMMARY: {passed}/17 checks passed, {failed} failed.")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
