// Does int8 weights actually buy bandwidth on the PRODUCTION access pattern?
//
// Established so far (bench_inproc_layer_cycling, real model):
//   custom cube : 20.55 ms same-layer -> 51.27 ms cycling 36 layers
//   aclnnMm     : 40.15 ms same-layer -> 42.90 ms cycling 36 layers
// Production decode.mlp measures 42.98 ms with aclnnMm, confirming decode is
// bound by COLD reads of distinct per-layer weights, at ~7 GB/s effective --
// about 6x short of the 43 GB/s device memcpy figure. Tiling cannot close that
// (aclnnMm is vendor-tuned and lands in the same place), so the only remaining
// lever is reading FEWER BYTES: int8 weights halve the MLP weight traffic.
//
// This measures the int8 MLP in the same two access patterns, on int8 weights
// derived from the REAL model weights (so magnitudes/quantization are honest),
// against the fp16 baseline in the same process. The question is not just "is
// int8 faster" but "does its advantage SURVIVE cold reads" -- the exact trap
// the custom cube kernel fell into.
#include "minicpmo/acl_context.h"
#include "minicpmo/language_model.h"
#include "minicpmo/quantized_weight.h"
#include "minicpmo/ops.h"
#include "minicpmo/tensor.h"
#include "minicpmo/weights.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace minicpmo;

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path = "./models/MiniCPM-o-4.5";
    if (argc > 1) model_path = argv[1];

    AclContext ctx(0);
    aclrtStream stream = ctx.stream();

    std::printf("Loading real model from %s ...\n", model_path.c_str());
    WeightsIndex index(model_path + "/model.safetensors");
    LanguageModelConfig lm_cfg = default_minicpmo45_lm_config();
    LanguageModelWeights lm_weights = load_language_model_weights(index, lm_cfg);
    index.release_shard_memory();
    std::printf("Model loaded.\n\n");

    std::vector<int> layers;
    for (size_t i = 0; i < lm_cfg.layer_types.size(); ++i) {
        if (lm_cfg.layer_types[i] != "full_attention") continue;
        const auto& w = lm_weights.layers[i];
        if (w.gate_w.shape().size() == 2 && w.gate_w.shape()[0] == 4096 &&
            w.gate_w.shape()[1] == 12288) {
            layers.push_back(static_cast<int>(i));
        }
    }
    const int n = static_cast<int>(layers.size());
    std::printf("Found %d full_attention layers with 4096x12288 MLP\n", n);
    if (n == 0) return 1;

    const int64_t K = 4096, N = 12288;

    Tensor x({1, K}, DType::Float16); x.allocate();
    Tensor gate({1, N}, DType::Float16); gate.allocate();
    Tensor up({1, N}, DType::Float16); up.allocate();
    Tensor gated({1, N}, DType::Float16); gated.allocate();
    Tensor out({1, K}, DType::Float16); out.allocate();
    std::vector<uint16_t> x_host(static_cast<size_t>(K), 0x3c00u);
    x.copy_from_host(x_host.data(), x_host.size() * sizeof(uint16_t));

    // --- fp16 baseline, both access patterns ---
    auto mlp_fp16 = [&](int layer) {
        const auto& w = lm_weights.layers[layer];
        matmul_b_transposed(x, w.gate_w, gate, stream);
        matmul_b_transposed(x, w.up_w, up, stream);
        silu_mul(gate, up, gated, stream);
        matmul_b_transposed(gated, w.down_w, out, stream);
    };

    for (int i = 0; i < 3; ++i) mlp_fp16(layers[0]);
    aclrtSynchronizeStream(stream);
    double t0 = now_ms();
    for (int i = 0; i < n; ++i) mlp_fp16(layers[0]);
    aclrtSynchronizeStream(stream);
    double fp16_same = (now_ms() - t0) / n;

    for (int i = 0; i < n; ++i) mlp_fp16(layers[i]);
    aclrtSynchronizeStream(stream);
    t0 = now_ms();
    for (int i = 0; i < n; ++i) mlp_fp16(layers[i]);
    aclrtSynchronizeStream(stream);
    double fp16_cycle = (now_ms() - t0) / n;

    std::printf("fp16  same=%8.4f ms   cycle=%8.4f ms  (%.2fx)\n",
                fp16_same, fp16_cycle, fp16_cycle / fp16_same);

    // --- int8 weights, quantized from the real fp16 weights ---
    // Per-layer int8 copies of gate/up/down, so cycling touches distinct bytes
    // exactly like fp16 cycling does, just half as many.
    std::printf("\nQuantizing %d layers to int8 (this takes a moment)...\n", n);
    struct Int8Layer { W8A8QuantizedWeight gate, up, down; };
    std::vector<Int8Layer> q(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const auto& w = lm_weights.layers[layers[i]];
        Int8Layer& ql = q[static_cast<size_t>(i)];
        ql.gate = quantize_dense_weight_w8a8(w.gate_w);
        ql.up   = quantize_dense_weight_w8a8(w.up_w);
        ql.down = quantize_dense_weight_w8a8(w.down_w);
    }
    aclrtSynchronizeStream(stream);
    std::printf("Quantized.\n\n");

    Tensor x_i8({1, K}, DType::Int8); x_i8.allocate();
    Tensor x_sc({1}, DType::Float16); x_sc.allocate();
    Tensor g_i8({1, N}, DType::Int8); g_i8.allocate();
    Tensor g_sc({1}, DType::Float16); g_sc.allocate();
    Tensor acc_n({1, N}, DType::Int32); acc_n.allocate();
    Tensor acc_k({1, K}, DType::Int32); acc_k.allocate();

    auto mlp_int8 = [&](int idx) {
        const Int8Layer& ql = q[static_cast<size_t>(idx)];
        w8a8_quantize(x, x_i8, x_sc, stream);
        matmul_w8a8_i32(x_i8, ql.gate.w_int8, acc_n, stream);
        w8a8_dequant(acc_n, x_sc, ql.gate.w_scale, gate, stream);
        matmul_w8a8_i32(x_i8, ql.up.w_int8, acc_n, stream);
        w8a8_dequant(acc_n, x_sc, ql.up.w_scale, up, stream);
        silu_mul(gate, up, gated, stream);
        w8a8_quantize(gated, g_i8, g_sc, stream);
        matmul_w8a8_i32(g_i8, ql.down.w_int8, acc_k, stream);
        w8a8_dequant(acc_k, g_sc, ql.down.w_scale, out, stream);
    };

    for (int i = 0; i < 3; ++i) mlp_int8(0);
    aclrtSynchronizeStream(stream);
    t0 = now_ms();
    for (int i = 0; i < n; ++i) mlp_int8(0);
    aclrtSynchronizeStream(stream);
    double i8_same = (now_ms() - t0) / n;

    for (int i = 0; i < n; ++i) mlp_int8(i);
    aclrtSynchronizeStream(stream);
    t0 = now_ms();
    for (int i = 0; i < n; ++i) mlp_int8(i);
    aclrtSynchronizeStream(stream);
    double i8_cycle = (now_ms() - t0) / n;

    std::printf("int8  same=%8.4f ms   cycle=%8.4f ms  (%.2fx)\n",
                i8_same, i8_cycle, i8_cycle / i8_same);

    // Effective bandwidth on the cycling (production) pattern.
    const double fp16_bytes = 3.0 * static_cast<double>(K) * static_cast<double>(N) * 2.0;
    const double i8_bytes   = 3.0 * static_cast<double>(K) * static_cast<double>(N) * 1.0;
    std::printf("\n=== cold-read (production) comparison ===\n");
    std::printf("fp16 : %8.4f ms for %.1f MB -> %5.2f GB/s\n",
                fp16_cycle, fp16_bytes / 1e6, fp16_bytes / (fp16_cycle * 1e6));
    std::printf("int8 : %8.4f ms for %.1f MB -> %5.2f GB/s\n",
                i8_cycle, i8_bytes / 1e6, i8_bytes / (i8_cycle * 1e6));
    std::printf("speedup on the pattern that matters: %.2fx\n", fp16_cycle / i8_cycle);
    if (i8_cycle < fp16_cycle * 0.75) {
        std::printf("-> int8 WINS on cold reads. Worth wiring into the decode path.\n");
    } else {
        std::printf("-> int8 does NOT win on cold reads; the win seen elsewhere was\n");
        std::printf("   a cache-hit artifact, same trap as the custom cube kernel.\n");
    }
    return 0;
}
