// End-to-end prefill + decode latency benchmark using the real production
// path (prefill_from_embeddings + decode_step_greedy), to measure the actual
// tokens/sec impact of the MatmulCubeCustom fast path rather than isolated
// matmul microbenchmarks.
#include "minicpmo/acl_context.h"
#include "minicpmo/decoder_layer.h"
#include "minicpmo/language_model.h"
#include "minicpmo/ops.h"
#include "minicpmo/profiling.h"
#include "minicpmo/tensor.h"
#include "minicpmo/weights.h"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
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
    int num_decode_steps = 32;
    if (argc > 2) num_decode_steps = std::atoi(argv[2]);

    std::vector<int32_t> input_ids = {151667, 198, 32313, 11, 279, 1196, 374,
                                       10161, 369, 279, 6722, 315, 9625, 13, 358};

    std::cout << "Loading model from " << model_path << " ..." << std::endl;
    AclContext ctx(0);

    WeightsIndex index(model_path + "/model.safetensors");
    LanguageModelConfig lm_cfg = default_minicpmo45_lm_config();
    LanguageModelWeights lm_weights = load_language_model_weights(index, lm_cfg);
    index.release_shard_memory();

    const int64_t max_seq = 4096;
    Tensor cos_table({max_seq, lm_cfg.rotary_dim / 2}, DType::Float16);
    Tensor sin_table({max_seq, lm_cfg.rotary_dim / 2}, DType::Float16);
    build_rope_tables(max_seq, lm_cfg, cos_table, sin_table);

    FullAttentionDecoderLayerConfig full_cfg{lm_cfg.num_q_heads, lm_cfg.num_kv_heads,
                                             lm_cfg.head_dim, lm_cfg.rotary_dim, lm_cfg.rms_epsilon};
    DecodeState state = make_decode_state(max_seq, lm_cfg.layer_types, full_cfg, ctx.stream());

    const int64_t T = static_cast<int64_t>(input_ids.size());
    Tensor prompt_hidden({T, lm_cfg.hidden_size}, DType::Float16); prompt_hidden.allocate();
    for (int64_t i = 0; i < T; ++i) {
        Tensor single_emb({1, lm_cfg.hidden_size}, DType::Float16); single_emb.allocate();
        embedding_lookup(lm_weights.embed, {input_ids[static_cast<size_t>(i)]}, single_emb, ctx.stream());
        int64_t offset = i * lm_cfg.hidden_size * static_cast<int64_t>(sizeof(uint16_t));
        aclrtMemcpyAsync(static_cast<char*>(prompt_hidden.data()) + offset,
                          lm_cfg.hidden_size * sizeof(uint16_t),
                          single_emb.data(), lm_cfg.hidden_size * sizeof(uint16_t),
                          ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream());
    }
    aclrtSynchronizeStream(ctx.stream());

    // Warm-up prefill run isn't meaningful (prefill is one-shot per session),
    // so just time the real prefill directly.
    double t0 = now_ms();
    Tensor last_hidden = prefill_from_embeddings(prompt_hidden, lm_weights, lm_cfg,
                                                 cos_table, sin_table, state, ctx.stream());
    aclrtSynchronizeStream(ctx.stream());
    double t1 = now_ms();
    double prefill_ms = t1 - t0;

    int64_t token = lm_head_greedy(last_hidden, lm_weights, lm_cfg, ctx.stream());

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Prefill: " << T << " tokens in " << prefill_ms << " ms ("
              << (static_cast<double>(T) / (prefill_ms / 1000.0)) << " tok/s)" << std::endl;

    // Decode loop: warm up a few steps (JIT/cache warm-up), then time.
    constexpr int kWarmup = 4;
    for (int i = 0; i < kWarmup && i < num_decode_steps; ++i) {
        token = decode_step_greedy(static_cast<int32_t>(token), lm_weights, lm_cfg,
                                   cos_table, sin_table, state, ctx.stream());
    }

    int timed_steps = num_decode_steps - kWarmup;
    if (timed_steps <= 0) timed_steps = num_decode_steps;

    profile_reset();
    double d0 = now_ms();
    for (int i = 0; i < timed_steps; ++i) {
        token = decode_step_greedy(static_cast<int32_t>(token), lm_weights, lm_cfg,
                                   cos_table, sin_table, state, ctx.stream());
    }
    double d1 = now_ms();
    double decode_ms = d1 - d0;
    double per_step_ms = decode_ms / timed_steps;

    std::cout << "Decode: " << timed_steps << " steps (after " << kWarmup
              << " warm-up) in " << decode_ms << " ms, "
              << per_step_ms << " ms/tok, "
              << (1000.0 / per_step_ms) << " tok/s" << std::endl;
    std::cout << "Final seq_len=" << state.seq_len << " last token=" << token << std::endl;
    profile_print();

    return 0;
}
