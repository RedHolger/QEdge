#include "backend/qnn_stub_backend.h"
#include "benchmark/benchmark.h"
#include "kv_cache/kv_cache.h"
#include "kv_cache/policy.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

// ---------------------------------------------------------------------------
// Power report table printer
// Called from main() after QNN benchmark finishes.
// ---------------------------------------------------------------------------

static void print_power_report_table(const std::vector<PowerReport>& reports) {
    std::cout << "\n";
    std::cout << "  QEdge Runtime — KV-cache CPU-stub report (Stage 01)\n";
    std::cout << "  Synthetic power estimates only; no Snapdragon/QNN device was used.\n";
    std::cout << "  Hardware status: unverified.\n\n";

    std::cout << std::left
              << std::setw(12) << "Policy"
              << std::setw(12) << "tok/s"
              << std::setw(12) << "Peak MB"
              << std::setw(10) << "Hit%"
              << std::setw(12) << "Est.mJ"
              << std::setw(12) << "tok/mJ"
              << "\n";
    std::cout << std::string(70, '-') << "\n";

    const PowerReport* best = nullptr;
    for (const auto& r : reports) {
        double peak_mb = static_cast<double>(r.peak_memory_bytes) / (1024.0 * 1024.0);

        std::cout << std::left
                  << std::setw(12) << r.policy_name
                  << std::setw(12) << std::fixed << std::setprecision(0) << r.tokens_per_sec
                  << std::setw(12) << std::fixed << std::setprecision(2) << peak_mb
                  << std::setw(10) << std::fixed << std::setprecision(1) << (r.hit_rate * 100.0f)
                  << std::setw(12) << std::fixed << std::setprecision(4) << r.estimated_mj
                  << std::setw(12) << std::fixed << std::setprecision(0) << r.tokens_per_mj
                  << "\n";

        if (!best || r.tokens_per_mj > best->tokens_per_mj) best = &r;
    }

    std::cout << std::string(70, '-') << "\n";

    if (best) {
        std::cout << "\n  CPU-stub comparison (not an NPU recommendation): "
                  << best->policy_name << " has the highest synthetic tok/mJ in this run ("
                  << std::fixed << std::setprecision(0) << best->tokens_per_mj << ").\n";
        std::cout << "    This is a CPU-stub estimate; no Snapdragon/Hexagon measurement exists.\n\n";
    }
}

// ---------------------------------------------------------------------------
// QNN benchmark runner — runs all 4 policies with QnnStubBackend
// ---------------------------------------------------------------------------

void run_qnn_power_benchmark(const ModelConfig& model_cfg, const RuntimeConfig& runtime_cfg) {
    std::cout << "\n[QnnStub] Running power-efficiency benchmark across all KV-cache policies...\n";

    QnnConfig qnn_cfg;
    qnn_cfg.sram_mb     = 4;
    qnn_cfg.verbose     = false;

    std::vector<EvictionPolicy> policies = {
        EvictionPolicy::FIFO,
        EvictionPolicy::LRU,
        EvictionPolicy::KeyDiff,
        EvictionPolicy::QUOKA,
    };
    std::vector<std::string> policy_names = { "FIFO", "LRU", "KeyDiff", "QUOKA" };

    Benchmark bench(model_cfg, runtime_cfg);
    std::vector<PowerReport> reports;

    for (size_t pi = 0; pi < policies.size(); ++pi) {
        QnnStubBackend qnn(qnn_cfg);
        qnn.reset_profile();

        RuntimeConfig rcfg = runtime_cfg;
        rcfg.policy = policies[pi];

        auto cache = create_kv_cache(model_cfg, policies[pi]);
        BenchResult br = bench.run_single(qnn, *cache);

        PowerReport pr = qnn.make_power_report(
            policy_names[pi], br.tokens_per_sec,
            cache->stats().peak_memory_bytes, cache->hit_rate());

        // Store actual tok/s in pr for table display
        // tokens_per_mj already set; we reconstruct tok/s from it for display
        reports.push_back(pr);

        std::cout << "  " << std::setw(10) << policy_names[pi]
                  << " tok/s=" << std::fixed << std::setprecision(0) << br.tokens_per_sec
                  << " hit=" << std::fixed << std::setprecision(1) << br.cache_hit_rate * 100 << "%"
                  << " evicts=" << br.evictions
                  << "\n";
    }

    print_power_report_table(reports);
}
