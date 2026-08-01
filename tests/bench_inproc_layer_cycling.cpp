// Final variable in the decode.mlp 51ms-vs-21ms investigation.
//
// Already ruled out (each proven in-process, with the real model resident):
//   - real vs synthetic weights           -> 21.4 vs 20.6 ms (no effect)
//   - activation magnitude 0.02 .. 1000   -> 20.4 .. 20.6 ms (no effect)
//   - fp16 saturation / denormals         -> no effect
//   - scratch allocation timing           -> no effect
//   - device memory footprint             -> no effect
//
// The one condition never reproduced: production touches a DIFFERENT layer's
// weights on every call, cycling through all 36 layers (~10.4 GB of distinct
// weight bytes per token). Every benchmark so far re-read the SAME ~288 MB
// weight set, which can stay resident in cache/DRAM row buffers across
// iterations. This measures the same MLP triple in two access patterns using
// the REAL weights:
//   A) hammer layer 0 repeatedly          (benchmark pattern)
//   B) walk layer 0,1,2,...,35 in order   (production pattern)
#include "minicpmo/acl_context.h"
#include "minicpmo/language_model.h"
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

    // Collect every full_attention layer whose MLP has the 4096/12288 shape.
    std::vector<int> layers;
    for (size_t i = 0; i < lm_cfg.layer_types.size(); ++i) {
        if (lm_cfg.layer_types[i] != "full_attention") continue;
        const auto& w = lm_weights.layers[i];
        if (w.gate_w.shape().size() == 2 && w.gate_w.shape()[0] == 4096 &&
            w.gate_w.shape()[1] == 12288) {
            layers.push_back(static_cast<int>(i));
        }
    }
    std::printf("Found %zu full_attention layers with 4096x12288 MLP\n", layers.size());
    if (layers.empty()) return 1;

    const int64_t K = 4096, N = 12288;
    Tensor x({1, K}, DType::Float16); x.allocate();
    Tensor gate({1, N}, DType::Float16); gate.allocate();
    Tensor up({1, N}, DType::Float16); up.allocate();
    Tensor gated({1, N}, DType::Float16); gated.allocate();
    Tensor out({1, K}, DType::Float16); out.allocate();
    std::vector<uint16_t> x_host(static_cast<size_t>(K), 0x3c00u);
    x.copy_from_host(x_host.data(), x_host.size() * sizeof(uint16_t));

    auto mlp_for = [&](int layer) {
        const auto& w = lm_weights.layers[layer];
        matmul_b_transposed(x, w.gate_w, gate, stream);
        matmul_b_transposed(x, w.up_w, up, stream);
        silu_mul(gate, up, gated, stream);
        matmul_b_transposed(gated, w.down_w, out, stream);
    };

    const int n = static_cast<int>(layers.size());

    // --- A: same layer, n times (what every prior benchmark did) ---
    for (int i = 0; i < 5; ++i) mlp_for(layers[0]);
    aclrtSynchronizeStream(stream);
    double t0 = now_ms();
    for (int i = 0; i < n; ++i) mlp_for(layers[0]);
    aclrtSynchronizeStream(stream);
    double t1 = now_ms();
    double same_ms = (t1 - t0) / n;
    std::printf("A) SAME layer x%d      : %8.4f ms/iter\n", n, same_ms);

    // --- B: walk all distinct layers once (what production does) ---
    for (int i = 0; i < n; ++i) mlp_for(layers[i]);  // warm
    aclrtSynchronizeStream(stream);
    double t2 = now_ms();
    for (int i = 0; i < n; ++i) mlp_for(layers[i]);
    aclrtSynchronizeStream(stream);
    double t3 = now_ms();
    double cycle_ms = (t3 - t2) / n;
    std::printf("B) CYCLE %d layers     : %8.4f ms/iter\n", n, cycle_ms);

    // --- A again, to rule out drift ---
    double t4 = now_ms();
    for (int i = 0; i < n; ++i) mlp_for(layers[0]);
    aclrtSynchronizeStream(stream);
    double t5 = now_ms();
    std::printf("A') SAME layer again   : %8.4f ms/iter\n", (t5 - t4) / n);

    std::printf("\n=== VERDICT ===\n");
    std::printf("same=%.2f ms  cycling=%.2f ms  ratio=%.2fx\n",
                same_ms, cycle_ms, cycle_ms / same_ms);
    if (cycle_ms > same_ms * 1.5) {
        std::printf("-> CONFIRMED: cost comes from touching DISTINCT weights per call.\n");
        std::printf("   Production is DRAM-bandwidth bound on cold weight reads; the\n");
        std::printf("   20ms benchmark number was an artifact of weight reuse.\n");
    } else {
        std::printf("-> Weight cycling does NOT explain the gap.\n");
    }
    return 0;
}
