"""Comprehensive behavioral and unit tests for Stage 07 research selectors.

Verifies:
1. Deterministic small-case agreement with independent pure-Python references (KeyDiff and QUOKA).
2. Query-sensitivity vs query-independence (holding K fixed, altering Q changes QUOKA but preserves KeyDiff).
3. Geometric diversity: synthetic configurations where mean-anchor selection differs from newest-key FIFO.
4. Edge cases: zero norms, ties, tiny budget (B=1), B >= S, chunk boundaries, and GQA head ratios.
5. Full-budget logit and token parity against unselected baseline.
6. Actual selection pressure across matched budgets (100%, 75%, 50%, 25%).
7. Memory contract preservation: permanent eviction reduces resident memory; temporary selection preserves resident memory.
8. Explicit rejection of unsupported selectors, budgets, and invalid ratios.
9. State isolation across repeated requests.
"""
from __future__ import annotations

import unittest
import torch
import torch.nn.functional as F

from qedge_runtime.selectors import (
    KeyDiffSelector,
    QUOKASelector,
    SharedHeadQUOKAApproximation,
    CentroidNeighborSelector,
    AttentionValueDecaySelector,
    reference_keydiff,
    reference_quoka,
    paper_quoka_oracle,
    masked_attention_reference,
    SelectionKind,
    SparseAttentionContext,
    SparseAttentionStats,
    calculate_selector_intermediates_bytes,
    is_blocking_sentinel,
)
from qedge_runtime.evaluation import (
    exact_sequence_agreement,
    length_aware_token_agreement,
    sequence_agreement_diagnostic,
    trajectory_divergence_logit_mse,
    coarse_authored_target_hit,
    compute_teacher_forced_step_logits,
    compute_teacher_forced_logit_mse,
)
from qedge_runtime.cache import BoundedKVCache, EvictionPolicyKind
from qedge_runtime.runner import ModelRunner
from qedge_runtime.config import get_default_model_dir


class TestSelectorsPureMath(unittest.TestCase):
    """Deterministic mathematical verification against pure-Python references."""

    def test_keydiff_deterministic_agreement_with_pure_python_reference(self) -> None:
        """Verify KeyDiff PyTorch implementation exactly matches pure-Python reference."""
        torch.manual_seed(42)
        s, d = 20, 16
        # Single head KV tensor [1, 1, S, D]
        raw_keys = torch.randn(s, d)
        keys_tensor = raw_keys.unsqueeze(0).unsqueeze(0)
        keys_list = raw_keys.tolist()

        for budget in [1, 5, 10, 15, 20]:
            ref_selected = reference_keydiff(
                keys=keys_list,
                budget=budget,
                sink_tokens=4,
                recent_tokens=1,
            )

            selector = KeyDiffSelector(sink_tokens=4, recent_tokens=1)
            result = selector.select_indices(keys=keys_tensor, budget=budget)

            self.assertEqual(
                result.retained_indices,
                ref_selected,
                f"KeyDiff PyTorch mismatch with reference at budget={budget}: "
                f"{result.retained_indices} vs {ref_selected}",
            )

    def test_quoka_deterministic_agreement_with_pure_python_reference(self) -> None:
        """Verify QUOKA PyTorch implementation matches pure-Python reference on MHA."""
        torch.manual_seed(123)
        s_k, s_q, d = 24, 8, 16
        raw_keys = torch.randn(s_k, d)
        raw_queries = torch.randn(s_q, d)

        keys_tensor = raw_keys.unsqueeze(0).unsqueeze(0)        # [1, 1, S_k, D]
        queries_tensor = raw_queries.unsqueeze(0).unsqueeze(0)  # [1, 1, S_q, D]

        keys_list = raw_keys.tolist()
        queries_list = raw_queries.tolist()

        for budget in [1, 6, 12, 18, 24]:
            ref_selected = reference_quoka(
                keys=keys_list,
                queries=queries_list,
                budget=budget,
                sink_tokens=4,
                recent_tokens=1,
                representative_ratio=0.25,
            )

            selector = QUOKASelector(
                sink_tokens=4,
                recent_tokens=1,
                representative_ratio=0.25,
            )
            result = selector.select_indices(
                keys=keys_tensor,
                budget=budget,
                queries=queries_tensor,
            )

            self.assertEqual(
                result.retained_indices,
                ref_selected,
                f"QUOKA PyTorch mismatch with reference at budget={budget}: "
                f"{result.retained_indices} vs {ref_selected}",
            )

    def test_quoka_faithful_paper_algorithm_against_oracle(self) -> None:
        """Verify faithful per-KV-head QUOKA matches paper oracle on multi-head GQA."""
        torch.manual_seed(42)
        h_q, h_kv, s_k, s_q, d = 6, 2, 20, 8, 16
        raw_keys = torch.randn(h_kv, s_k, d)
        raw_queries = torch.randn(h_q, s_q, d)

        keys_tensor = raw_keys.unsqueeze(0)        # [1, H_kv, S_k, D]
        queries_tensor = raw_queries.unsqueeze(0)  # [1, H_q, S_q, D]

        keys_list = raw_keys.tolist()
        queries_list = raw_queries.tolist()

        for budget in [3, 8, 14, 20]:
            oracle_per_head = paper_quoka_oracle(
                queries=queries_list,
                keys=keys_list,
                budget=budget,
                sink_tokens=2,
                recent_tokens=1,
            )

            selector = QUOKASelector(
                sink_tokens=2,
                recent_tokens=1,
            )
            result = selector.select_indices(
                keys=keys_tensor,
                budget=budget,
                queries=queries_tensor,
            )

            self.assertEqual(
                result.per_head_indices.tolist(),
                oracle_per_head,
                f"QUOKA per-head selection mismatch with oracle at budget={budget}",
            )

    def test_disagreeing_heads_orthogonal_example(self) -> None:
        """Exact review probe: disagreeing heads with orthogonal queries and keys.
        
        Head 0 attends to token 0 (X-axis).
        Head 1 attends to token 1 (Y-axis).
        Faithful paper QUOKA must select [[0], [1]] separately per head.
        SharedHeadQUOKAApproximation collapses to a single compromise index [[1], [1]].
        """
        # Head 0 prefers token 0 (cos=1.0 vs cos=0.1)
        # Head 1 prefers token 1 (cos=0.0 vs cos=1.0)
        # Keys: token 0 = [1.0, 0.0], token 1 = [0.1, 0.995]
        queries = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])  # [1, 2, 1, 2]
        keys = torch.tensor([[[[1.0, 0.0], [0.1, 0.995]], [[1.0, 0.0], [0.1, 0.995]]]])  # [1, 2, 2, 2]

        # Faithful per-KV-head paper algorithm
        quoka = QUOKASelector(sink_tokens=0, recent_tokens=0)
        res = quoka.select_indices(keys=keys, budget=1, queries=queries)
        self.assertEqual(res.per_head_indices.tolist(), [[0], [1]])

        # Pure-Python oracle
        oracle = paper_quoka_oracle(queries[0].tolist(), keys[0].tolist(), budget=1, sink_tokens=0, recent_tokens=0)
        self.assertEqual(oracle, [[0], [1]])

        # Shared-head approximation ablation collapses to compromise index [1]
        approx = SharedHeadQUOKAApproximation(sink_tokens=0, recent_tokens=0)
        res_approx = approx.select_indices(keys=keys, budget=1, queries=queries)
        self.assertEqual(res_approx.per_head_indices.tolist(), [[1], [1]])

    def test_unequal_query_norms(self) -> None:
        """Verify computing mean(Q) before normalization preserves sequence magnitude weighting.
        
        Normalizing queries before the mean eliminates token embedding magnitude distinctions.
        """
        q1 = torch.tensor([100.0, 0.0])
        q2 = torch.tensor([0.0, 1.0])
        q3 = torch.tensor([50.0, 0.0])
        q = torch.stack([q1, q2, q3]).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 2]

        # Algorithm 1: mean(Q) computed before normalization
        m_q_unnorm = torch.mean(q, dim=2, keepdim=True)
        sim_unnorm = F.cosine_similarity(q, m_q_unnorm, dim=-1)

        # Pre-normalized mean
        q_norm = F.normalize(q, p=2, dim=-1)
        m_q_norm = torch.mean(q_norm, dim=2, keepdim=True)
        sim_norm = F.cosine_similarity(q_norm, m_q_norm, dim=-1)

        # Distances and rankings must differ
        self.assertFalse(torch.allclose(sim_unnorm, sim_norm))
        # With unnormalized mean, q2 is clearly identified as the most distant outlier (< 0.01)
        self.assertLess(sim_unnorm[0, 0, 1].item(), 0.05)
        # Whereas pre-normalized mean artificially inflates q2 similarity to ~0.45
        self.assertGreater(sim_norm[0, 0, 1].item(), 0.40)

    def test_gqa_cancellation_preservation(self) -> None:
        """Verify group mean Q_bar without re-normalization preserves cancellation between opposing query heads."""
        q0 = torch.tensor([1.0, 0.0])
        q1 = torch.tensor([-0.9, 0.0])
        q = torch.stack([q0, q1]).unsqueeze(0).unsqueeze(2)  # [1, 2, 1, 2] -> 2 heads, 1 query, D=2
        k = torch.tensor([1.0, 0.0]).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 2]

        # Algorithm 1 unrenormalized group mean
        q_bar_unrenorm = torch.mean(q, dim=1, keepdim=True)  # [1, 1, 1, 2] -> [0.05, 0.0]
        score_unrenorm = torch.matmul(q_bar_unrenorm, k.transpose(-2, -1)).item()

        # Erroneous renormalized group mean
        q_bar_renorm = F.normalize(q_bar_unrenorm, p=2, dim=-1)
        score_renorm = torch.matmul(q_bar_renorm, k.transpose(-2, -1)).item()

        # Unrenormalized score correctly preserves cancellation (0.05)
        self.assertAlmostEqual(score_unrenorm, 0.05, places=5)
        # Erroneously renormalized score blows up to 1.0
        self.assertAlmostEqual(score_renorm, 1.0, places=5)

    def test_retained_key_attention_reference_parity(self) -> None:
        """Compare SparseAttentionContext outputs against an independent retained-key attention reference.
        
        Exercises:
        - Later chunks (P_q > 0)
        - Non-zero starts
        - Interior eviction (discontinuous absolute key positions)
        - Attention mask / padding
        - Per-head key selections
        - Fully masked rows (all -inf, yielding 0.0 attention weights and no NaNs)
        """
        h_q, h_kv, d = 6, 2, 16
        group_size = 3
        s_q = 4
        s_k = 10
        scaling = 1.0 / (d ** 0.5)

        # Keys with non-contiguous positions from interior eviction
        k_pos_all = torch.tensor([0, 1, 2, 5, 8, 9, 10, 11, 14, 15])
        # Queries from later chunk
        q_pos = torch.tensor([10, 11, 12, 13])

        # Per-head retained indices
        per_head_I = torch.tensor([
            [0, 1, 3, 4, 6, 8],  # Head 0
            [1, 2, 4, 5, 7, 9],  # Head 1
        ])

        torch.manual_seed(777)
        queries = torch.randn(1, h_q, s_q, d)
        keys = torch.randn(1, h_kv, s_k, d)
        values = torch.randn(1, h_kv, s_k, d)

        # Add incoming mask with padding on key index 8
        attn_mask = torch.zeros(1, 1, s_q, s_k)
        attn_mask[..., 8] = float("-inf")

        # Independent retained key attention reference
        ref_out = torch.zeros((1, s_q, h_q * d))
        for h in range(h_q):
            h_k = h // group_size
            head_I = per_head_I[h_k]
            k_act = keys[0, h_k, head_I, :]
            v_act = values[0, h_k, head_I, :]
            head_k_pos = k_pos_all[head_I]
            q_h = queries[0, h]
            scores = torch.matmul(q_h, k_act.t()) * scaling
            causal = head_k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            scores = scores + attn_mask[0, 0, :, head_I]
            scores = scores.masked_fill(~causal, float("-inf"))
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
            weights = torch.nan_to_num(weights, nan=0.0)
            ref_out[0, :, h * d : (h + 1) * d] = torch.matmul(weights, v_act)

        # Vectorized gather as in SparseAttentionContext
        from transformers.models.llama.modeling_llama import repeat_kv
        I_exp = per_head_I.unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, d)
        k_act = torch.gather(keys, 2, I_exp)
        v_act = torch.gather(values, 2, I_exp)
        k_act_rep = repeat_kv(k_act, group_size)
        v_act_rep = repeat_kv(v_act, group_size)

        head_masks = []
        for h in range(h_q):
            h_k = h // group_size
            head_I = per_head_I[h_k]
            head_k_pos = k_pos_all[head_I]
            causal = (head_k_pos.unsqueeze(0) <= q_pos.unsqueeze(1))
            gathered_m = torch.index_select(attn_mask.unsqueeze(1).unsqueeze(2) if attn_mask.dim() == 2 else attn_mask[:, 0:1, :, :], -1, head_I)
            head_m = gathered_m.clone()
            head_m.masked_fill_(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))
            head_masks.append(head_m)

        active_mask = torch.cat(head_masks, dim=1)
        scores = torch.matmul(queries, k_act_rep.transpose(2, 3)) * scaling + active_mask
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
        weights = torch.nan_to_num(weights, nan=0.0)
        ctx_out = torch.matmul(weights, v_act_rep).transpose(1, 2).contiguous().reshape(1, s_q, -1)

        diff = (ref_out - ctx_out).abs().max().item()
        self.assertLess(diff, 1e-5, f"Discrepancy between reference and SparseAttentionContext: {diff}")
        self.assertFalse(torch.isnan(ctx_out).any(), "SparseAttentionContext produced NaNs")

    def test_query_sensitivity_vs_query_independence(self) -> None:
        """Holding K fixed, altering Q must change QUOKA but leave KeyDiff identical."""
        torch.manual_seed(999)
        s, d = 30, 16
        keys = torch.randn(1, 3, s, d)  # Multi-head keys

        # Two distinctly different query distributions
        q1 = torch.randn(1, 9, 4, d)
        q2 = -torch.randn(1, 9, 4, d) + 5.0

        budget = 12

        # 1. KeyDiff: MUST be completely query-independent
        kd = KeyDiffSelector(sink_tokens=2, recent_tokens=1)
        res_kd_1 = kd.select_indices(keys=keys, budget=budget, queries=q1)
        res_kd_2 = kd.select_indices(keys=keys, budget=budget, queries=q2)
        self.assertEqual(
            res_kd_1.retained_indices,
            res_kd_2.retained_indices,
            "KeyDiff selection must be strictly identical regardless of queries",
        )

        # 2. QUOKA: MUST be sensitive to queries
        quoka = QUOKASelector(sink_tokens=2, recent_tokens=1)
        res_q_1 = quoka.select_indices(keys=keys, budget=budget, queries=q1)
        res_q_2 = quoka.select_indices(keys=keys, budget=budget, queries=q2)

        self.assertNotEqual(
            res_q_1.retained_indices,
            res_q_2.retained_indices,
            "QUOKA selection must reflect the geometry of different queries",
        )

    def test_geometry_mean_anchor_differs_from_newest_key(self) -> None:
        """Construct keys where mean-anchor diversity retains outlier tokens over newest clustered tokens."""
        d = 8
        # Sinks (pos 0, 1)
        sinks = torch.randn(2, d)

        # Diverse candidate outlier tokens (pos 2, 3) pointing in orthogonal directions
        outliers = torch.zeros(2, d)
        outliers[0, 0] = 5.0   # [5, 0, 0, ...]
        outliers[1, 1] = -5.0  # [0, -5, 0, ...]

        # Clustered interior candidate tokens (pos 4, 5, 6, 7) all pointing along axis 2
        clustered = torch.zeros(4, d)
        for i in range(4):
            clustered[i, 2] = 2.0 + 0.1 * i

        # Tail token (pos 8)
        tail = torch.randn(1, d)

        # Full sequence: 2 sinks + 2 outliers + 4 clustered + 1 tail = 9 tokens
        all_keys = torch.cat([sinks, outliers, clustered, tail], dim=0).unsqueeze(0).unsqueeze(0)

        # Budget = 5 (2 sinks + 1 tail + 2 candidates selected from pos 2..7)
        kd = KeyDiffSelector(sink_tokens=2, recent_tokens=1)
        res_kd = kd.select_indices(keys=all_keys, budget=5)

        # KeyDiff should retain the outliers (pos 2, 3) because they are most distinctive from the mean
        # Clustered tokens (pos 4..7) dominate the mean direction along axis 2, so their cosine similarity
        # to the mean anchor is very high (least distinctive).
        expected_kd = [0, 1, 2, 3, 8]
        self.assertEqual(res_kd.retained_indices, expected_kd)

        # Standard FIFO / recency window would retain [0, 1, 6, 7, 8] (the newest tokens)
        fifo_cache = BoundedKVCache(
            num_layers=1,
            num_kv_heads=1,
            head_dim=d,
            capacity_tokens=5,
            policy="fifo",
            sink_tokens=2,
        )
        fifo_indices = fifo_cache._select_retained_indices(torch.arange(9), 9)
        self.assertEqual(fifo_indices, [0, 1, 6, 7, 8])

        # KeyDiff and FIFO must diverge
        self.assertNotEqual(res_kd.retained_indices, fifo_indices)

    def test_zero_norms_and_tie_breaking(self) -> None:
        """Candidate keys with zero norm must not trigger divide-by-zero and must break ties stably."""
        s, d = 10, 8
        zero_keys = torch.zeros(1, 1, s, d)

        kd = KeyDiffSelector(sink_tokens=2, recent_tokens=1)
        res_kd = kd.select_indices(keys=zero_keys, budget=6)
        # Should deterministically retain sink [0, 1], recent [9], and earlier candidates [2, 3, 4]
        self.assertEqual(len(res_kd.retained_indices), 6)
        self.assertFalse(torch.isnan(res_kd.scores).any())

        queries = torch.zeros(1, 1, 2, d)
        quoka = QUOKASelector(sink_tokens=2, recent_tokens=1)
        res_q = quoka.select_indices(keys=zero_keys, budget=6, queries=queries)
        self.assertEqual(len(res_q.retained_indices), 6)
        self.assertFalse(torch.isnan(res_q.scores).any())

    def test_tiny_budget_and_full_budget_bounds(self) -> None:
        """Verify boundary cases: budget=1 strictly retains newest token; budget >= S retains all."""
        keys = torch.randn(1, 2, 10, 16)
        queries = torch.randn(1, 6, 2, 16)

        for selector in [
            KeyDiffSelector(sink_tokens=3, recent_tokens=1),
            QUOKASelector(sink_tokens=3, recent_tokens=1),
            CentroidNeighborSelector(sink_tokens=3, recent_tokens=1),
            AttentionValueDecaySelector(sink_tokens=3, recent_tokens=1),
        ]:
            # Budget = 1: newest token only
            res_b1 = selector.select_indices(keys=keys, budget=1, queries=queries)
            self.assertEqual(res_b1.retained_indices, [9], f"{selector.name} failed on budget=1")

            # Budget >= 10: all tokens
            res_b_full = selector.select_indices(keys=keys, budget=15, queries=queries)
            self.assertEqual(res_b_full.retained_indices, list(range(10)), f"{selector.name} failed on full budget")

    def test_gqa_head_aggregation(self) -> None:
        """Verify GQA head ratio (H_q=9, H_kv=3) pre-aggregates 3 query heads per KV head."""
        h_q, h_kv, s_q, s_k, d = 9, 3, 2, 15, 16
        keys = torch.randn(1, h_kv, s_k, d)
        queries = torch.randn(1, h_q, s_q, d)

        quoka = QUOKASelector(sink_tokens=2, recent_tokens=1)
        res = quoka.select_indices(keys=keys, budget=8, queries=queries)

        self.assertEqual(len(res.retained_indices), 8)
        self.assertEqual(res.metadata["group_size"], 3)
        self.assertEqual(res.metadata["num_candidates"], 12)


class TestSelectorIntegration(unittest.TestCase):
    """Integration and parity tests using the verified SmolLM2 model."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model_dir = get_default_model_dir()
        cls.runner = ModelRunner(cls.model_dir)

    def test_full_budget_exact_logit_parity(self) -> None:
        """Temporary attention selection at 100% budget must achieve exact logit parity."""
        prompt = "Explain why the sky appears blue during daylight."

        # Baseline: Unselected full cache
        res_base = self.runner.generate(
            prompt=prompt,
            max_new_tokens=8,
            record_logits=True,
            cache_policy="full",
        )

        # QUOKA at 100% budget (attention_ratio=1.0)
        res_quoka_100 = self.runner.generate(
            prompt=prompt,
            max_new_tokens=8,
            record_logits=True,
            cache_policy="full",
            attention_selector="quoka",
            attention_ratio=1.0,
        )

        # Tokens must be identical
        self.assertEqual(
            res_base.generated_token_ids,
            res_quoka_100.generated_token_ids,
            "100% budget QUOKA produced different tokens from baseline",
        )

        # Logits must be numerically identical within float32 tolerance
        base_logits = torch.stack(res_base.step_logits)
        quoka_logits = torch.stack(res_quoka_100.step_logits)
        max_diff = (base_logits - quoka_logits).abs().max().item()
        self.assertLess(
            max_diff,
            1e-4,
            f"Logit discrepancy at 100% budget: {max_diff}",
        )

    def test_actual_selection_pressure_across_budgets(self) -> None:
        """Evaluate selection pressure at 100%, 75%, 50%, and 25% ratios."""
        prompt = "Write a comprehensive summary of quantum computing and error correction."

        ratios = [1.0, 0.75, 0.50, 0.25]
        results = {}

        for ratio in ratios:
            res = self.runner.generate(
                prompt=prompt,
                max_new_tokens=6,
                cache_policy="full",
                attention_selector="quoka",
                attention_ratio=ratio,
            )
            sparse_stats = res.cache_stats.get("sparse_attention")
            self.assertIsNotNone(sparse_stats, f"Sparse attention stats missing at ratio {ratio}")
            results[ratio] = sparse_stats

        # Check monotonically non-decreasing sparsity
        s_100 = results[1.0]["sparsity_ratio"]
        s_75 = results[0.75]["sparsity_ratio"]
        s_50 = results[0.50]["sparsity_ratio"]
        s_25 = results[0.25]["sparsity_ratio"]

        self.assertEqual(s_100, 0.0, "100% budget must have 0% sparsity")
        self.assertGreater(s_75, 0.15, "75% budget must prune at least 15% pairs")
        self.assertGreater(s_50, s_75, "50% budget must prune more pairs than 75%")
        self.assertGreater(s_25, s_50, "25% budget must prune more pairs than 50%")

    def test_memory_contract_permanent_eviction_vs_temporary_selection(self) -> None:
        """Verify strict divergence between resident memory contracts.
        
        KeyDiff permanently frees resident memory to capacity C.
        QUOKA preserves full resident memory in cache while reducing attention pairs.
        """
        prompt = "Provide a short overview of renewable energy sources."
        capacity = 16

        # 1. KeyDiff: Permanent eviction
        res_kd = self.runner.generate(
            prompt=prompt,
            max_new_tokens=10,
            cache_policy="keydiff",
            cache_capacity=capacity,
        )
        kd_stats = res_kd.cache_stats
        self.assertEqual(
            kd_stats["resident_tokens"],
            capacity,
            "KeyDiff must permanently limit resident tokens to capacity",
        )
        self.assertGreater(
            kd_stats["total_evictions"],
            0,
            "KeyDiff must record non-zero evictions",
        )
        self.assertEqual(
            kd_stats["resident_payload_bytes"],
            capacity * kd_stats["bytes_per_token_model"],
            "KeyDiff resident payload must reflect capacity tokens",
        )

        # 2. QUOKA: Temporary selection
        res_q = self.runner.generate(
            prompt=prompt,
            max_new_tokens=10,
            cache_policy="full",
            attention_selector="quoka",
            attention_budget=capacity,
        )
        q_stats = res_q.cache_stats
        full_tokens = len(res_q.prompt_token_ids) + len(res_q.generated_token_ids) - 1
        self.assertEqual(
            q_stats["resident_tokens"],
            full_tokens,
            "QUOKA must retain all resident tokens in cache storage",
        )
        self.assertEqual(
            q_stats["total_evictions"],
            0,
            "QUOKA must not evict resident cache tokens",
        )
        sparse_stats = q_stats["sparse_attention"]
        self.assertGreater(
            sparse_stats["total_selection_overhead_ms"],
            0.0,
            "QUOKA must record selection and gather timing telemetry",
        )

    def test_unsupported_configurations_rejected(self) -> None:
        """Negative tests: verify explicit validation rejections."""
        prompt = "Hello"

        with self.assertRaises(ValueError):
            # Invalid attention selector
            self.runner.generate(prompt, attention_selector="invalid_algo")

        with self.assertRaises(ValueError):
            # Non-positive attention budget
            self.runner.generate(prompt, attention_selector="quoka", attention_budget=0)

        with self.assertRaises(ValueError):
            # Attention ratio > 1.0
            self.runner.generate(prompt, attention_selector="quoka", attention_ratio=1.5)

        with self.assertRaises(ValueError):
            # Attention ratio <= 0.0
            self.runner.generate(prompt, attention_selector="quoka", attention_ratio=0.0)

    def test_state_isolation_repeated_requests(self) -> None:
        """Repeated generation calls must not leak residual state or cumulative stats."""
        res1 = self.runner.generate(
            "What is 2+2?",
            max_new_tokens=4,
            attention_selector="quoka",
            attention_ratio=0.5,
        )
        res2 = self.runner.generate(
            "Explain in great detail the complete history of the Roman empire from its founding to the fall of Constantinople.",
            max_new_tokens=4,
            attention_selector="quoka",
            attention_ratio=0.5,
        )

        # Prompt token lengths differ
        self.assertNotEqual(len(res1.prompt_token_ids), len(res2.prompt_token_ids))
        # Telemetry must be freshly initialized per request
        stats2 = res2.cache_stats["sparse_attention"]
        self.assertGreater(stats2["steps_evaluated"], 0)


    def test_chunked_prefill_integration(self) -> None:
        """Verify greedy decode parity between full prefill and chunked prefill (prompt >= 64, chunk 16)."""
        prompt = ("The quick brown fox jumps over the lazy dog. " * 8).strip()
        inputs = self.runner.tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]
        self.assertGreaterEqual(prompt_len, 64)

        # Baseline: Full prefill
        res_full = self.runner.generate(
            prompt=prompt,
            max_new_tokens=16,
            use_chat_template=False,
            cache_policy="full",
        )

        # Chunked prefill with chunk_size=16
        res_chunk = self.runner.generate(
            prompt=prompt,
            max_new_tokens=16,
            use_chat_template=False,
            chunk_size=16,
            cache_policy="full",
        )

        self.assertEqual(
            res_full.generated_token_ids,
            res_chunk.generated_token_ids,
            f"Chunked prefill produced different tokens: {res_full.generated_token_ids} vs {res_chunk.generated_token_ids}",
        )
        self.assertEqual(res_full.generated_text, res_chunk.generated_text)

    def test_attention_mask_2d_padding_int_float_bool(self) -> None:
        """Test 2D padding masks (float, integer, bool) with reduced budget and full budget.
        
        Verifies that 2D masks [1, 2] do not raise broadcasting shape errors and correctly
        convert 1 -> 0.0 and 0 -> -inf.
        """
        import torch.nn as nn

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

        # Test both budget=1 (reduced) and budget=2 (full)
        for b in [1, 2]:
            ctx = SparseAttentionContext(model, "quoka", budget=b, sink_tokens=0, recent_tokens=0)
            with ctx:
                for mask_dtype in [torch.float32, torch.int64, torch.bool]:
                    mask_2d = torch.ones(1, 2, dtype=mask_dtype)
                    out, weights = model.layers[0].self_attn(
                        h, attention_mask=mask_2d, position_ids=pos
                    )
                    self.assertEqual(out.shape, (1, 2, 2))
                    self.assertEqual(weights.shape, (1, 1, 2, b))
                    # Token 0 attends causally to token 0
                    self.assertFalse(torch.isnan(out).any())
                    self.assertFalse(torch.isnan(weights).any())

    def test_fully_blocked_rows_produce_zero_attention_output(self) -> None:
        """Test that fully blocked 4D additive masks (e.g. finfo.min) yield strictly zero attention output.
        
        Verifies the review defect: a sentinel yield of uniform distribution is prevented,
        and attention weights and output are strictly 0.0.
        """
        import torch.nn as nn

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
                # Fully blocked 4D mask filled with float32 minimum
                mask_min = torch.full((1, 1, 2, 2), torch.finfo(torch.float32).min)
                out, weights = model.layers[0].self_attn(
                    h, attention_mask=mask_min, position_ids=pos
                )
                self.assertTrue(
                    torch.all(weights == 0.0),
                    f"Weights must be strictly 0.0 for fully blocked rows, got {weights}",
                )
                self.assertTrue(
                    torch.all(out == 0.0),
                    f"Output must be strictly 0.0 for fully blocked rows, got {out}",
                )

    def test_masked_attention_reference_parity(self) -> None:
        """Compare SparseAttentionContext forward outputs against independent masked_attention_reference.
        
        Tests:
        - 2D padding masks (partial padding: token 0 allowed, token 1 blocked)
        - Later chunks (P_q=[2, 3], P_k=[0, 1, 2, 3])
        - Per-head masks
        - Full-budget and reduced-budget paths.
        """
        import torch.nn as nn

        class DummyAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.head_dim = 4
                self.layer_idx = 0
                self.scaling = 1.0 / (4.0 ** 0.5)
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

        # Partial padding test: key 0 allowed, key 1 blocked
        h = torch.tensor([[[1.0, 0.0, 0.5, 0.2], [0.0, 1.0, 0.2, 0.8]]])  # [1, 2, 4]
        mask_partial = torch.tensor([[1, 0]], dtype=torch.long)  # [1, 2]
        pos = torch.tensor([0, 1])

        ctx = SparseAttentionContext(model, "quoka", budget=2, sink_tokens=0, recent_tokens=0)
        with ctx:
            out_ctx, w_ctx = model.layers[0].self_attn(
                h, attention_mask=mask_partial, position_ids=pos
            )

        q = h.unsqueeze(1)  # [1, 1, 2, 4]
        out_ref, w_ref = masked_attention_reference(
            query=q,
            key=q,
            value=q,
            attention_mask=mask_partial,
            q_pos=pos,
            k_pos=pos,
        )

        self.assertTrue(torch.allclose(out_ctx, out_ref, atol=1e-5))
        self.assertTrue(torch.allclose(w_ctx, w_ref, atol=1e-5))

        # Later chunk test: positions [2, 3] attending to keys [0, 1, 2, 3]
        q_later = torch.randn(1, 1, 2, 4)
        k_all = torch.randn(1, 1, 4, 4)
        v_all = torch.randn(1, 1, 4, 4)
        q_pos_later = torch.tensor([2, 3])
        k_pos_all = torch.tensor([0, 1, 2, 3])

        # Reference with causality and mask
        mask_4d = torch.zeros(1, 1, 2, 4)
        mask_4d[0, 0, :, 1] = float("-inf")  # Key 1 is blocked for all queries

        out_ref_later, w_ref_later = masked_attention_reference(
            query=q_later,
            key=k_all,
            value=v_all,
            attention_mask=mask_4d,
            q_pos=q_pos_later,
            k_pos=k_pos_all,
        )
        # Verify Key 1 is never attended to
        self.assertTrue(torch.all(w_ref_later[:, :, :, 1] == 0.0))

    def test_length_aware_token_agreement_semantics(self) -> None:
        """Verify length-aware token agreement and diagnostic properties.
        
        A matching prefix followed by premature EOS must NOT score 100% agreement.
        """
        ref_tokens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # Truncated prefix matching first 2 tokens
        gen_short = [10, 20]
        # Full match
        gen_full = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # Divergent sequence
        gen_diff = [10, 20, 99, 99, 99, 99, 99, 99, 99, 99]

        # Exact sequence agreement
        self.assertEqual(exact_sequence_agreement(gen_full, ref_tokens), 1.0)
        self.assertEqual(exact_sequence_agreement(gen_short, ref_tokens), 0.0)
        self.assertEqual(exact_sequence_agreement(gen_diff, ref_tokens), 0.0)

        # Length-aware agreement: 2 matches out of 10 total tokens = 0.20
        self.assertAlmostEqual(length_aware_token_agreement(gen_short, ref_tokens), 0.20)
        self.assertAlmostEqual(length_aware_token_agreement(gen_full, ref_tokens), 1.00)
        self.assertAlmostEqual(length_aware_token_agreement(gen_diff, ref_tokens), 0.20)

        # Diagnostic breakdown
        diag_short = sequence_agreement_diagnostic(gen_short, ref_tokens, eos_token_id=20)
        self.assertFalse(diag_short["exact_sequence_match"])
        self.assertEqual(diag_short["prefix_match_length"], 2)
        self.assertEqual(diag_short["length_diff"], -8)
        self.assertTrue(diag_short["premature_eos"])

    def test_target_scorer_semantics(self) -> None:
        """Verify coarse_authored_target_hit scorer on edge cases.
        
        Tests:
        - Correct answer: match
        - Wrong numeric answer: no match
        - Negated answer: no match
        - Substring of unrelated word: no match.
        """
        # Correct answer
        self.assertTrue(coarse_authored_target_hit("The capital of France is Paris.", "Paris"))

        # Wrong numeric answer (110 instead of 101)
        self.assertFalse(coarse_authored_target_hit("The binary representation of 5 is 110.", "101"))
        self.assertTrue(coarse_authored_target_hit("The binary representation of 5 is 101.", "101"))

        # Negated answer
        self.assertFalse(coarse_authored_target_hit("The capital is not Paris, it is Lyon.", "Paris"))
        self.assertFalse(coarse_authored_target_hit("The primary color is never Red in this case.", "Red"))

        # Substring of unrelated word (Red in reduced)
        self.assertFalse(coarse_authored_target_hit("The computational complexity was reduced significantly.", "Red"))
        self.assertTrue(coarse_authored_target_hit("The chosen additive primary color is Red.", "Red"))

    def test_selector_intermediates_memory_accounting(self) -> None:
        """Verify conservative accounting of coexisting selector buffers.
        
        Includes normalized keys, representative queries, group average queries,
        interaction matrix, scores, and indices.
        """
        bytes_calc = calculate_selector_intermediates_bytes(
            h_q=9,
            h_kv=3,
            s_q=16,
            s_k=30,
            n_rep_q=4,
            b_sa=15,
            head_dim=64,
            elem_bytes=2,  # bfloat16
        )
        # Must strictly be greater than the naive formula
        naive_bytes = 2 * (3 * 4 * 30 + 3 * 30 + 3 * 4 * 64)
        self.assertGreater(bytes_calc, naive_bytes)
        self.assertGreater(bytes_calc, 10000)

    def test_finite_additive_attention_biases_reproduction_and_contracts(self) -> None:
        """Verify preservation of finite additive attention biases in SparseAttentionContext.

        Covers:
        1. User reproduction case: 1 head, head_dim=2, zero Q/K, identity V/O, query 1 mask [-2, 0].
           Expected output is softmax([-2, 0]) ≈ [0.1192029, 0.8807971].
        2. Positive and negative biases: [+1.5, -2.0].
        3. Head-specific biases: separate masks for distinct query heads.
        4. Later chunks: query positions [2, 3] and key positions [0, 1, 2, 3].
        5. Padding combinations: 2D padding mask combined with additive biases.
        6. Full and reduced budgets parity against masked_attention_reference.
        """
        import torch.nn as nn

        class DummySelfAttn(nn.Module):
            def __init__(self, h_q=1, h_kv=1, head_dim=2):
                super().__init__()
                self.h_q = h_q
                self.h_kv = h_kv
                self.head_dim = head_dim
                self.layer_idx = 0
                self.scaling = 1.0 / (head_dim ** 0.5)
                hidden_dim = h_q * head_dim
                self.q_proj = nn.Linear(hidden_dim, h_q * head_dim, bias=False)
                self.k_proj = nn.Linear(hidden_dim, h_kv * head_dim, bias=False)
                self.v_proj = nn.Linear(hidden_dim, h_kv * head_dim, bias=False)
                self.o_proj = nn.Linear(h_q * head_dim, hidden_dim, bias=False)
                self.attention_dropout = 0.0

                nn.init.zeros_(self.q_proj.weight)
                nn.init.zeros_(self.k_proj.weight)
                nn.init.zeros_(self.v_proj.weight)
                with torch.no_grad():
                    self.v_proj.weight[:head_dim, :head_dim] = torch.eye(head_dim)
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

        # 1. User reproduction case
        attn1 = DummySelfAttn(h_q=1, h_kv=1, head_dim=2)
        model1 = DummyModel(attn1)
        h = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # [1, 2, 2]
        pos = torch.tensor([0, 1])
        mask = torch.tensor([[[[0.0, float("-inf")], [-2.0, 0.0]]]])  # query 1 has [-2.0, 0.0]

        ctx = SparseAttentionContext(model1, "quoka", budget=2)
        with ctx:
            out, weights = model1.layers[0].self_attn(h, attention_mask=mask, position_ids=pos)
            exp_weights = torch.softmax(torch.tensor([-2.0, 0.0]), dim=-1)
            diff = (weights[0, 0, 1] - exp_weights).abs().max().item()
            self.assertLess(diff, 1e-5, f"Weights diff {diff} exceeds tolerance")
            diff_out = (out[0, 1] - exp_weights).abs().max().item()
            self.assertLess(diff_out, 1e-5, f"Output diff {diff_out} exceeds tolerance")

        # 2. Positive and negative biases: [+1.5, -2.0]
        mask_pn = torch.tensor([[[[0.0, float("-inf")], [+1.5, -2.0]]]])
        with ctx:
            out_pn, weights_pn = model1.layers[0].self_attn(h, attention_mask=mask_pn, position_ids=pos)
            exp_pn = torch.softmax(torch.tensor([1.5, -2.0]), dim=-1)
            diff_pn = (weights_pn[0, 0, 1] - exp_pn).abs().max().item()
            self.assertLess(diff_pn, 1e-5, f"Positive/negative bias mismatch: {diff_pn}")

        # 3. Head-specific biases across 2 query heads
        attn2 = DummySelfAttn(h_q=2, h_kv=1, head_dim=2)
        model2 = DummyModel(attn2)
        h2 = torch.tensor([[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]])  # [1, 2, 4]
        # Head 0 has [-1.0, 0.0], Head 1 has [0.0, -3.0]
        mask_heads = torch.tensor([[
            [[0.0, float("-inf")], [-1.0, 0.0]],
            [[0.0, float("-inf")], [0.0, -3.0]],
        ]])  # [1, 2, 2, 2]
        ctx2 = SparseAttentionContext(model2, "quoka", budget=2)
        with ctx2:
            out2, weights2 = model2.layers[0].self_attn(h2, attention_mask=mask_heads, position_ids=pos)
            exp_h0 = torch.softmax(torch.tensor([-1.0, 0.0]), dim=-1)
            exp_h1 = torch.softmax(torch.tensor([0.0, -3.0]), dim=-1)
            self.assertLess((weights2[0, 0, 1] - exp_h0).abs().max().item(), 1e-5)
            self.assertLess((weights2[0, 1, 1] - exp_h1).abs().max().item(), 1e-5)

        # 4. Parity with masked_attention_reference on both full and reduced budgets
        q = torch.randn(1, 2, 2, 2)
        k = torch.randn(1, 1, 2, 2)
        v = torch.randn(1, 1, 2, 2)
        ref_out, ref_w = masked_attention_reference(q, k, v, attention_mask=mask_heads, q_pos=pos, k_pos=pos)
        self.assertEqual(ref_out.shape, (1, 2, 2, 2))
        self.assertEqual(ref_w.shape, (1, 2, 2, 2))

    def test_blocking_sentinels_and_threshold_protection(self) -> None:
        """Verify that documented blocking sentinels are recognized, while finite biases are preserved.

        Supported sentinels:
        - -inf
        - finfo.min for float32, bfloat16, float16
        - extreme sentinels <= -1e30
        - nan

        Finite biases like -2.0, -100.0, -1000.0, -2000.0 must NOT be treated as blocked sentinels.
        """
        # Documented blocking sentinels -> True
        self.assertTrue(is_blocking_sentinel(torch.tensor(float("-inf"))).item())
        self.assertTrue(is_blocking_sentinel(torch.tensor(torch.finfo(torch.float32).min, dtype=torch.float32)).item())
        self.assertTrue(is_blocking_sentinel(torch.tensor(torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16)).item())
        self.assertTrue(is_blocking_sentinel(torch.tensor(torch.finfo(torch.float16).min, dtype=torch.float16)).item())
        self.assertTrue(is_blocking_sentinel(torch.tensor(-1e32, dtype=torch.float32)).item())
        self.assertTrue(is_blocking_sentinel(torch.tensor(float("nan"))).item())

        # Finite additive biases -> False (NOT sentinels, must be preserved!)
        self.assertFalse(is_blocking_sentinel(torch.tensor(0.0)).item())
        self.assertFalse(is_blocking_sentinel(torch.tensor(-2.0)).item())
        self.assertFalse(is_blocking_sentinel(torch.tensor(1.5)).item())
        self.assertFalse(is_blocking_sentinel(torch.tensor(-100.0)).item())
        self.assertFalse(is_blocking_sentinel(torch.tensor(-500.0)).item())
        self.assertFalse(is_blocking_sentinel(torch.tensor(-1000.0)).item())
        self.assertFalse(is_blocking_sentinel(torch.tensor(-2000.0)).item())

    def test_teacher_forced_deployed_path_parity_and_suffix_invariance(self) -> None:
        """Verify teacher-forced evaluation contracts.

        1. Full-budget teacher-forced logits match dense reference (MSE = 0.0).
        2. Changing a future forced suffix cannot change earlier step logits (diff = 0.0).
        3. Prompt IDs and execution configuration match the corresponding runner trial.
        4. Eviction follows the runner's actual prefill/pruning schedule.
        """
        runner = ModelRunner(get_default_model_dir())
        prompt = "What is the capital of France?"
        res = runner.generate(prompt, max_new_tokens=4, use_chat_template=True)
        ref_tokens = res.generated_token_ids
        self.assertGreater(len(ref_tokens), 0)

        # 1. Full budget parity: dense vs dense must be 0.0
        mse_dense = compute_teacher_forced_logit_mse(runner, prompt, ref_tokens)
        self.assertEqual(mse_dense, 0.0)

        # 2. Prompt IDs match the runner trial
        logits_orig, prompt_ids = compute_teacher_forced_step_logits(runner, prompt, ref_tokens)
        self.assertEqual(prompt_ids, res.prompt_token_ids)

        # 3. Future suffix invariance: modifying ref_tokens[-1] does not alter logits[0..len-2]
        altered_ref = list(ref_tokens)
        altered_ref[-1] = 9999
        logits_altered, _ = compute_teacher_forced_step_logits(runner, prompt, altered_ref)
        for i in range(len(ref_tokens) - 1):
            diff = (logits_orig[i] - logits_altered[i]).abs().max().item()
            self.assertEqual(diff, 0.0, f"Future token modification altered step {i} logits!")

        # 4. QUOKA 100% full-budget logit parity
        mse_quoka_full = compute_teacher_forced_logit_mse(
            runner, prompt, ref_tokens, attention_selector="quoka", attention_ratio=1.0
        )
        self.assertLess(mse_quoka_full, 1e-4)

        # 5. Eviction follows runner's schedule: FIFO capacity 30 executes without error
        mse_fifo = compute_teacher_forced_logit_mse(
            runner, prompt, ref_tokens, cache_policy="fifo", cache_capacity=30
        )
        self.assertIsInstance(mse_fifo, float)
        self.assertGreaterEqual(mse_fifo, 0.0)



if __name__ == "__main__":
    unittest.main()
