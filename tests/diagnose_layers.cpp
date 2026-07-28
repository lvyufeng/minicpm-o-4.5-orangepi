// Standalone diagnostic tool: loads the model, runs prefill on a fixed
// token sequence, and prints per-layer hidden-state statistics (mean abs,
// max abs, any NaN/Inf) to help locate where the forward pass diverges.
#include "minicpmo/acl_context.h"
#include "minicpmo/decoder_layer.h"
#include "minicpmo/language_model.h"
#include "minicpmo/ops.h"
#include "minicpmo/tensor.h"
#include "minicpmo/weights.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace minicpmo;

namespace {

float f16_to_f32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        bits = sign;
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

void print_stats(const char* label, const Tensor& t) {
    const size_t n = t.numel();
    std::vector<uint16_t> host(n);
    t.copy_to_host(host.data(), host.size() * sizeof(uint16_t));
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    int nan_count = 0, inf_count = 0;
    for (size_t i = 0; i < n; ++i) {
        float v = f16_to_f32(host[i]);
        if (std::isnan(v)) { ++nan_count; continue; }
        if (std::isinf(v)) { ++inf_count; continue; }
        float a = std::fabs(v);
        sum_abs += a;
        if (a > max_abs) max_abs = a;
    }
    std::cout << std::fixed << std::setprecision(5)
              << "  " << label
              << " mean|x|=" << (sum_abs / static_cast<double>(n))
              << " max|x|=" << max_abs
              << " nan=" << nan_count << " inf=" << inf_count
              << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path = "./models/MiniCPM-o-4.5";
    if (argc > 1) model_path = argv[1];

    std::vector<int32_t> input_ids = {16, 11, 220, 17, 11, 220, 18, 11, 220, 19};
    if (argc > 2) {
        input_ids.clear();
        for (int i = 2; i < argc; ++i) input_ids.push_back(std::atoi(argv[i]));
    }

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

    std::vector<std::string> layer_types = lm_cfg.layer_types;
    FullAttentionDecoderLayerConfig full_cfg{lm_cfg.num_q_heads, lm_cfg.num_kv_heads,
                                             lm_cfg.head_dim, lm_cfg.rotary_dim, lm_cfg.rms_epsilon};
    DecodeState state = make_decode_state(max_seq, layer_types, full_cfg, ctx.stream());

    const int64_t T = static_cast<int64_t>(input_ids.size());
    std::cout << "Prefilling " << T << " tokens: [";
    for (auto id : input_ids) std::cout << id << " ";
    std::cout << "]" << std::endl;

    // Build prompt_hidden via embedding lookup, same as backend_server does.
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

    print_stats("embed input", prompt_hidden);

    // Manually replicate prefill_from_embeddings but print stats after each layer.
    Tensor hidden({T, lm_cfg.hidden_size}, DType::Float16); hidden.allocate();
    Tensor next({T, lm_cfg.hidden_size}, DType::Float16); next.allocate();
    check_acl(aclrtMemcpyAsync(hidden.data(), hidden.size_bytes(), prompt_hidden.data(),
                               prompt_hidden.size_bytes(), ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()),
              "copy prompt_hidden");
    aclrtSynchronizeStream(ctx.stream());

    std::vector<int32_t> row_to_t(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) row_to_t[t] = static_cast<int32_t>(t);

    int full_i = 0;
    for (int64_t layer = 0; layer < lm_cfg.num_layers; ++layer) {
        const auto& lw = lm_weights.layers[layer];
        FullAttentionDecoderLayerWeights ww{
            &lw.input_norm_w, &lw.post_norm_w, &lw.q_w, &lw.k_w, &lw.v_w,
            &lw.o_w, &lw.q_norm_w, &lw.k_norm_w, &lw.gate_w, &lw.up_w, &lw.down_w,
        };
        full_attention_decoder_layer_with_cache(hidden, ww, cos_table, sin_table,
                                                row_to_t, full_cfg, state.full[full_i], next, ctx.stream());
        ++full_i;
        check_acl(aclrtMemcpyAsync(hidden.data(), hidden.size_bytes(), next.data(),
                                   next.size_bytes(), ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()),
                  "advance hidden");
        aclrtSynchronizeStream(ctx.stream());

        if (layer < 3 || layer % 6 == 0 || layer == lm_cfg.num_layers - 1) {
            std::ostringstream label;
            label << "layer " << layer;
            print_stats(label.str().c_str(), hidden);
        }
    }
    state.seq_len = T;

    Tensor last_hidden({1, lm_cfg.hidden_size}, DType::Float16); last_hidden.allocate();
    check_acl(aclrtMemcpy(last_hidden.data(), last_hidden.size_bytes(),
                          static_cast<const uint8_t*>(hidden.data()) + (T - 1) * lm_cfg.hidden_size * sizeof(uint16_t),
                          last_hidden.size_bytes(), ACL_MEMCPY_DEVICE_TO_DEVICE),
              "extract last hidden");

    print_stats("final hidden (last token)", last_hidden);

    int64_t token = lm_head_greedy(last_hidden, lm_weights, lm_cfg, ctx.stream());
    std::cout << "\nArgmax next token: " << token << std::endl;

    return 0;
}
