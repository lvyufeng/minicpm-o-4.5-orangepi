#include "minicpmo/decoder_layer.h"

#include "minicpmo/acl_context.h"
#include "minicpmo/ops.h"
#include "minicpmo/profiling.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace minicpmo {
namespace {

bool w8a8_decode_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("MINICPM_W8A8_DECODE");
        return v != nullptr && std::string(v) != "0" && std::string(v) != "false";
    }();
    return enabled;
}

uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

void check_ptr(const Tensor* t, const char* name) {
    if (t == nullptr) {
        throw std::runtime_error(std::string("missing decoder layer weight: ") + name);
    }
}

void copy_col_block(const Tensor& src, int64_t col_offset, Tensor& dst, aclrtStream stream) {
    const int64_t rows = src.shape()[0];
    const int64_t src_cols = src.shape()[1];
    const int64_t dst_cols = dst.shape()[1];
    const size_t elem = dtype_size(src.dtype());
    const size_t src_row_bytes = static_cast<size_t>(src_cols) * elem;
    const size_t dst_row_bytes = static_cast<size_t>(dst_cols) * elem;
    const size_t block_bytes = static_cast<size_t>(dst_cols) * elem;
    auto* s = static_cast<const uint8_t*>(src.data());
    auto* d = static_cast<uint8_t*>(dst.data());
    for (int64_t r = 0; r < rows; ++r) {
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(r) * dst_row_bytes, block_bytes,
                                   s + static_cast<size_t>(r) * src_row_bytes + static_cast<size_t>(col_offset) * elem,
                                   block_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_col_block");
    }
    // Remove sync - let operations queue asynchronously
}

void copy_head_to_seq(const Tensor& src_heads, int64_t head, int64_t heads_per_token,
                      Tensor& dst_seq, aclrtStream stream) {
    const int64_t tokens = dst_seq.shape()[0];
    const int64_t dim = dst_seq.shape()[1];
    const size_t row_bytes = static_cast<size_t>(dim) * dtype_size(src_heads.dtype());
    auto* s = static_cast<const uint8_t*>(src_heads.data());
    auto* d = static_cast<uint8_t*>(dst_seq.data());
    for (int64_t t = 0; t < tokens; ++t) {
        const int64_t src_row = t * heads_per_token + head;
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(t) * row_bytes, row_bytes,
                                   s + static_cast<size_t>(src_row) * row_bytes, row_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_head_to_seq");
    }
    // Remove sync - let operations queue asynchronously
}

void copy_seq_to_head_block(const Tensor& src_seq, Tensor& dst, int64_t col_offset,
                            aclrtStream stream) {
    const int64_t rows = src_seq.shape()[0];
    const int64_t dst_cols = dst.shape()[1];
    const int64_t src_cols = src_seq.shape()[1];
    const size_t elem = dtype_size(src_seq.dtype());
    const size_t src_row_bytes = static_cast<size_t>(src_cols) * elem;
    const size_t dst_row_bytes = static_cast<size_t>(dst_cols) * elem;
    auto* s = static_cast<const uint8_t*>(src_seq.data());
    auto* d = static_cast<uint8_t*>(dst.data());
    for (int64_t r = 0; r < rows; ++r) {
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(r) * dst_row_bytes + static_cast<size_t>(col_offset) * elem,
                                   src_row_bytes,
                                   s + static_cast<size_t>(r) * src_row_bytes,
                                   src_row_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_seq_to_head_block");
    }
    // Remove sync - let operations queue asynchronously
}

void copy_heads_from_cols(const Tensor& src, int64_t heads, int64_t head_dim,
                          Tensor& dst, aclrtStream stream) {
    const int64_t rows = src.shape()[0];
    const int64_t src_cols = src.shape()[1];
    const size_t elem = dtype_size(src.dtype());
    const size_t src_row_bytes = static_cast<size_t>(src_cols) * elem;
    const size_t head_bytes = static_cast<size_t>(head_dim) * elem;
    auto* s = static_cast<const uint8_t*>(src.data());
    auto* d = static_cast<uint8_t*>(dst.data());
    for (int64_t t = 0; t < rows; ++t) {
        for (int64_t h = 0; h < heads; ++h) {
            check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(t * heads + h) * head_bytes, head_bytes,
                                       s + static_cast<size_t>(t) * src_row_bytes + static_cast<size_t>(h * head_dim) * elem,
                                       head_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                      "copy_heads_from_cols");
        }
    }
    check_acl(aclrtSynchronizeStream(stream), "copy_heads_from_cols sync");
}

void split_q_gate(const Tensor& src, int64_t heads, int64_t head_dim,
                  Tensor& q_out, Tensor& gate_out, aclrtStream stream) {
    const int64_t rows = src.shape()[0];
    const size_t elem = dtype_size(src.dtype());
    const size_t src_row_bytes = static_cast<size_t>(src.shape()[1]) * elem;
    const size_t head_bytes = static_cast<size_t>(head_dim) * elem;
    const size_t q_row_bytes = static_cast<size_t>(q_out.shape()[1]) * elem;
    auto* s = static_cast<const uint8_t*>(src.data());
    auto* dq = static_cast<uint8_t*>(q_out.data());
    auto* dg = static_cast<uint8_t*>(gate_out.data());
    for (int64_t t = 0; t < rows; ++t) {
        for (int64_t h = 0; h < heads; ++h) {
            const size_t src_off = static_cast<size_t>(t) * src_row_bytes
                                 + static_cast<size_t>(h * 2 * head_dim) * elem;
            const size_t dst_off = static_cast<size_t>(t) * q_row_bytes
                                 + static_cast<size_t>(h * head_dim) * elem;
            check_acl(aclrtMemcpyAsync(dq + dst_off, head_bytes, s + src_off, head_bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                      "split_q_gate q");
            check_acl(aclrtMemcpyAsync(dg + dst_off, head_bytes, s + src_off + head_bytes, head_bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                      "split_q_gate gate");
        }
    }
    check_acl(aclrtSynchronizeStream(stream), "split_q_gate sync");
}


void copy_cache_head_to_seq(const Tensor& cache, int64_t head, int64_t heads_per_token,
                            int64_t head_dim, int64_t rows, Tensor& dst_seq,
                            aclrtStream stream) {
    const size_t elem = dtype_size(cache.dtype());
    const size_t cache_row_bytes = static_cast<size_t>(cache.shape()[1]) * elem;
    const size_t head_bytes = static_cast<size_t>(head_dim) * elem;
    auto* s = static_cast<const uint8_t*>(cache.data());
    auto* d = static_cast<uint8_t*>(dst_seq.data());
    for (int64_t r = 0; r < rows; ++r) {
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(r) * head_bytes, head_bytes,
                                   s + static_cast<size_t>(r) * cache_row_bytes + static_cast<size_t>(head * head_dim) * elem,
                                   head_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_cache_head_to_seq");
    }
    (void)heads_per_token;
    check_acl(aclrtSynchronizeStream(stream), "copy_cache_head_to_seq sync");
}

void copy_matrix_rows(const Tensor& src, int64_t src_row, Tensor& dst, int64_t dst_row,
                      int64_t rows, aclrtStream stream) {
    const size_t row_bytes = static_cast<size_t>(src.shape()[1]) * dtype_size(src.dtype());
    auto* s = static_cast<const uint8_t*>(src.data());
    auto* d = static_cast<uint8_t*>(dst.data());
    for (int64_t r = 0; r < rows; ++r) {
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(dst_row + r) * row_bytes, row_bytes,
                                   s + static_cast<size_t>(src_row + r) * row_bytes, row_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_matrix_rows");
    }
    check_acl(aclrtSynchronizeStream(stream), "copy_matrix_rows sync");
}


// Matmul weights can be stored as [N, K] (legacy matmul_b_transposed layout)
// or [K, N] (pre-transposed for cube fast path); accept either.
inline bool matmul_shape_ok(const Tensor* t, int64_t N, int64_t K) {
    const auto& s = t->shape();
    return s == std::vector<int64_t>{N, K} || s == std::vector<int64_t>{K, N};
}

bool w8a8_weight_ready(const W8A8QuantizedWeight* w8_weight) {
    return w8a8_decode_enabled() && w8_weight != nullptr && w8_weight->w_int8.data() != nullptr;
}

void matmul_decode_w8a8_prequant(const Tensor& x_int8,
                                 const Tensor& x_scale,
                                 const W8A8QuantizedWeight& w8_weight,
                                 Tensor& out,
                                 aclrtStream stream) {
    Tensor acc({1, w8_weight.N}, DType::Int32); acc.allocate();
    matmul_w8a8_i32(x_int8, w8_weight.w_int8, acc, stream);
    w8a8_dequant(acc, x_scale, w8_weight.w_scale, out, stream);
}

void matmul_decode_dispatch(const Tensor& x,
                            const Tensor* dense_weight,
                            const W4A16QuantizedWeight* quant_weight,
                            const W8A8QuantizedWeight* w8_weight,
                            Tensor& out,
                            aclrtStream stream) {
    if (w8a8_weight_ready(w8_weight) && x.shape().size() == 2 && x.shape()[0] == 1) {
        Tensor x_int8(x.shape(), DType::Int8); x_int8.allocate();
        Tensor x_scale({1}, DType::Float16); x_scale.allocate();
        w8a8_quantize(x, x_int8, x_scale, stream);
        matmul_decode_w8a8_prequant(x_int8, x_scale, *w8_weight, out, stream);
        return;
    }
    if (quant_weight != nullptr && x.shape().size() == 2 && x.shape()[0] == 1) {
        matmul_w4a16(x, quant_weight->w_int8, quant_weight->scales, out, stream);
        return;
    }
    matmul_b_transposed(x, *dense_weight, out, stream);
}

void validate_shapes(const Tensor& hidden,
                     const FullAttentionDecoderLayerWeights& w,
                     const FullAttentionDecoderLayerConfig& c,
                     const Tensor& out) {
    check_ptr(w.input_norm_weight, "input_norm_weight");
    check_ptr(w.post_attention_norm_weight, "post_attention_norm_weight");
    check_ptr(w.q_proj_weight, "q_proj_weight");
    check_ptr(w.k_proj_weight, "k_proj_weight");
    check_ptr(w.v_proj_weight, "v_proj_weight");
    check_ptr(w.o_proj_weight, "o_proj_weight");
    check_ptr(w.q_norm_weight, "q_norm_weight");
    check_ptr(w.k_norm_weight, "k_norm_weight");
    check_ptr(w.gate_proj_weight, "gate_proj_weight");
    check_ptr(w.up_proj_weight, "up_proj_weight");
    check_ptr(w.down_proj_weight, "down_proj_weight");

    if (hidden.shape().size() != 2 || out.shape() != hidden.shape()) {
        throw std::runtime_error("decoder layer hidden/out must be [T, H] and same shape");
    }
    if (hidden.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("decoder layer requires fp16 hidden/out");
    }
    if (c.num_q_heads <= 0 || c.num_kv_heads <= 0 || c.head_dim <= 0 || c.rotary_dim <= 0) {
        throw std::runtime_error("decoder layer invalid config dims");
    }
    if (c.num_q_heads % c.num_kv_heads != 0) {
        throw std::runtime_error("decoder layer num_q_heads must be divisible by num_kv_heads");
    }
}

}  // namespace

namespace {

void run_full_attention_core(const Tensor& hidden,
                             const FullAttentionDecoderLayerWeights& weights,
                             const Tensor& cos_table,
                             const Tensor& sin_table,
                             const std::vector<int32_t>& row_to_t,
                             const FullAttentionDecoderLayerConfig& config,
                             FullAttentionLayerCache* cache,
                             int64_t cache_offset,
                             Tensor& out,
                             aclrtStream stream) {
    validate_shapes(hidden, weights, config, out);

    const int64_t T = hidden.shape()[0];
    const int64_t Hidden = hidden.shape()[1];
    const int64_t NumQHeads = config.num_q_heads;
    const int64_t NumKVHeads = config.num_kv_heads;
    const int64_t QPerKV = NumQHeads / NumKVHeads;
    const int64_t HeadDim = config.head_dim;
    const int64_t QMainDim = NumQHeads * HeadDim;
    const int64_t KVDim = NumKVHeads * HeadDim;

    // Detect if the model uses gated attention by checking q_proj shape
    const auto& q_shape = weights.q_proj_weight->shape();
    int64_t q_proj_out = 0;
    if (q_shape.size() == 2) {
        if (q_shape[1] == Hidden) q_proj_out = q_shape[0];
        else if (q_shape[0] == Hidden) q_proj_out = q_shape[1];
    }
    const bool use_gated_attn = (q_proj_out == QMainDim * 2);
    const int64_t QProjOut = use_gated_attn ? (QMainDim * 2) : QMainDim;

    const int64_t Intermediate = [&]{
        const auto& s = weights.gate_proj_weight->shape();
        if (s.size() == 2) {
            if (s[1] == Hidden) return s[0];
            if (s[0] == Hidden) return s[1];
        }
        return s[0];
    }();

    auto check_shape = [&](const char* name, const Tensor* t, const std::vector<int64_t>& expected) {
        if (t->shape() != expected) {
            std::ostringstream oss;
            oss << "decoder layer weight shape mismatch: " << name << " expected [";
            for (size_t i = 0; i < expected.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << expected[i];
            }
            oss << "], got [";
            for (size_t i = 0; i < t->shape().size(); ++i) {
                if (i > 0) oss << ", ";
                oss << t->shape()[i];
            }
            oss << "]";
            throw std::runtime_error(oss.str());
        }
    };

    check_shape("input_norm_weight", weights.input_norm_weight, {Hidden});
    check_shape("post_attention_norm_weight", weights.post_attention_norm_weight, {Hidden});

    auto check_matmul = [&](const char* name, const Tensor* t, int64_t N, int64_t K) {
        if (!matmul_shape_ok(t, N, K)) {
            std::ostringstream oss;
            oss << "decoder layer " << name << " shape invalid for matmul: expected ["
                << N << ", " << K << "] or [" << K << ", " << N << "], got [";
            for (size_t i = 0; i < t->shape().size(); ++i) {
                if (i > 0) oss << ", ";
                oss << t->shape()[i];
            }
            oss << "]";
            throw std::runtime_error(oss.str());
        }
    };

    check_matmul("q_proj_weight", weights.q_proj_weight, QProjOut, Hidden);
    check_matmul("k_proj_weight", weights.k_proj_weight, KVDim, Hidden);
    check_matmul("v_proj_weight", weights.v_proj_weight, KVDim, Hidden);
    check_matmul("o_proj_weight", weights.o_proj_weight, Hidden, QMainDim);

    check_shape("q_norm_weight", weights.q_norm_weight, {HeadDim});
    check_shape("k_norm_weight", weights.k_norm_weight, {HeadDim});

    check_matmul("gate_proj_weight", weights.gate_proj_weight, Intermediate, Hidden);
    check_matmul("up_proj_weight", weights.up_proj_weight, Intermediate, Hidden);
    check_matmul("down_proj_weight", weights.down_proj_weight, Hidden, Intermediate);
    if (static_cast<int64_t>(row_to_t.size()) != T) {
        throw std::runtime_error("decoder layer row_to_t size must match sequence length");
    }
    if (cache != nullptr) {
        // Cache is [num_kv_heads, max_seq, head_dim]
        if (cache->k_cache.shape().size() != 3 ||
            cache->k_cache.shape()[0] != NumKVHeads ||
            cache->k_cache.shape()[2] != HeadDim ||
            cache->v_cache.shape() != cache->k_cache.shape()) {
            throw std::runtime_error("full attention cache shape mismatch");
        }
        if (cache_offset + T > cache->k_cache.shape()[1]) {
            throw std::runtime_error("full attention cache overflow");
        }
    }

    Tensor normed({T, Hidden}, DType::Float16); normed.allocate();
    rms_norm(hidden, *weights.input_norm_weight, normed, config.rms_epsilon, stream);

    Tensor q_full({T, QProjOut}, DType::Float16); q_full.allocate();
    Tensor k_full({T, KVDim}, DType::Float16); k_full.allocate();
    Tensor v_full({T, KVDim}, DType::Float16); v_full.allocate();
    matmul_b_transposed(normed, *weights.q_proj_weight, q_full, stream);
    matmul_b_transposed(normed, *weights.k_proj_weight, k_full, stream);
    matmul_b_transposed(normed, *weights.v_proj_weight, v_full, stream);

    Tensor q_only({T, QMainDim}, DType::Float16); q_only.allocate();
    Tensor q_gate({T, QMainDim}, DType::Float16);  // Only allocate if needed
    if (use_gated_attn) {
        q_gate.allocate();
        split_q_gate(q_full, NumQHeads, HeadDim, q_only, q_gate, stream);
    } else {
        // No gating: q_full is already QMainDim, just copy to q_only
        check_acl(aclrtMemcpyAsync(q_only.data(), q_only.size_bytes(),
                                   q_full.data(), q_full.size_bytes(),
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "q_full -> q_only");
        check_acl(aclrtSynchronizeStream(stream), "q_only copy sync");
    }

    Tensor q_heads({T * NumQHeads, HeadDim}, DType::Float16); q_heads.allocate();
    Tensor k_heads({T * NumKVHeads, HeadDim}, DType::Float16); k_heads.allocate();
    copy_heads_from_cols(q_only, NumQHeads, HeadDim, q_heads, stream);
    copy_heads_from_cols(k_full, NumKVHeads, HeadDim, k_heads, stream);

    Tensor q_normed({T * NumQHeads, HeadDim}, DType::Float16); q_normed.allocate();
    Tensor k_normed({T * NumKVHeads, HeadDim}, DType::Float16); k_normed.allocate();
    rms_norm(q_heads, *weights.q_norm_weight, q_normed, config.rms_epsilon, stream);
    rms_norm(k_heads, *weights.k_norm_weight, k_normed, config.rms_epsilon, stream);

    std::vector<int32_t> q_row_to_t(T * NumQHeads);
    std::vector<int32_t> k_row_to_t(T * NumKVHeads);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < NumQHeads; ++h) q_row_to_t[t * NumQHeads + h] = row_to_t[t];
        for (int64_t h = 0; h < NumKVHeads; ++h) k_row_to_t[t * NumKVHeads + h] = row_to_t[t];
    }
    Tensor q_rope({T * NumQHeads, HeadDim}, DType::Float16); q_rope.allocate();
    Tensor k_rope({T * NumKVHeads, HeadDim}, DType::Float16); k_rope.allocate();
    apply_rope_partial(q_normed, cos_table, sin_table, q_row_to_t, config.rotary_dim, q_rope, stream);
    apply_rope_partial(k_normed, cos_table, sin_table, k_row_to_t, config.rotary_dim, k_rope, stream);

    if (cache != nullptr) {
        // Cache is now [num_kv_heads, max_seq, head_dim]. Write K/V for timesteps
        // [cache_offset, cache_offset+T) by scattering each head's T rows into its plane.
        const size_t elem = dtype_size(k_rope.dtype());
        const size_t head_bytes = static_cast<size_t>(HeadDim) * elem;
        const size_t plane_stride = static_cast<size_t>(cache->k_cache.shape()[1] * HeadDim) * elem;

        // k_rope is [T*NumKVHeads, HeadDim]; gather head h's T rows and write to plane h.
        auto* k_src = static_cast<const uint8_t*>(k_rope.data());
        auto* k_base = static_cast<uint8_t*>(cache->k_cache.data());
        for (int64_t h = 0; h < NumKVHeads; ++h) {
            auto* k_plane = k_base + h * plane_stride + static_cast<size_t>(cache_offset) * head_bytes;
            for (int64_t t = 0; t < T; ++t) {
                check_acl(aclrtMemcpyAsync(k_plane + t * head_bytes, head_bytes,
                                           k_src + static_cast<size_t>(t * NumKVHeads + h) * head_bytes, head_bytes,
                                           ACL_MEMCPY_DEVICE_TO_DEVICE, stream), "k prefill scatter");
            }
        }

        // v_full is [T, NumKVHeads*HeadDim]; unpack and scatter each head's T rows.
        auto* v_src = static_cast<const uint8_t*>(v_full.data());
        auto* v_base = static_cast<uint8_t*>(cache->v_cache.data());
        const size_t v_row_bytes = static_cast<size_t>(KVDim) * elem;
        for (int64_t h = 0; h < NumKVHeads; ++h) {
            auto* v_plane = v_base + h * plane_stride + static_cast<size_t>(cache_offset) * head_bytes;
            for (int64_t t = 0; t < T; ++t) {
                check_acl(aclrtMemcpyAsync(v_plane + t * head_bytes, head_bytes,
                                           v_src + static_cast<size_t>(t) * v_row_bytes + h * head_bytes, head_bytes,
                                           ACL_MEMCPY_DEVICE_TO_DEVICE, stream), "v prefill scatter");
            }
        }
        check_acl(aclrtSynchronizeStream(stream), "kv cache write sync");
    }

    Tensor scale({T, T}, DType::Float16);
    std::vector<uint16_t> scale_host(static_cast<size_t>(T * T), f32_to_f16_bits(1.0f / std::sqrt(static_cast<float>(HeadDim))));
    scale.copy_from_host(scale_host.data(), scale_host.size() * sizeof(uint16_t));

    std::vector<uint16_t> mask_host(static_cast<size_t>(T * T));
    for (int64_t r = 0; r < T; ++r) {
        for (int64_t c = 0; c < T; ++c) {
            mask_host[r * T + c] = f32_to_f16_bits(row_to_t[c] <= row_to_t[r] ? 0.0f : -65504.0f);
        }
    }
    Tensor causal_mask({T, T}, DType::Float16);
    causal_mask.copy_from_host(mask_host.data(), mask_host.size() * sizeof(uint16_t));

    Tensor attn_out({T, QMainDim}, DType::Float16); attn_out.allocate();
    Tensor q_seq({T, HeadDim}, DType::Float16); q_seq.allocate();
    Tensor k_seq({T, HeadDim}, DType::Float16); k_seq.allocate();
    Tensor v_seq({T, HeadDim}, DType::Float16); v_seq.allocate();
    Tensor scores({T, T}, DType::Float16); scores.allocate();
    Tensor scaled_scores({T, T}, DType::Float16); scaled_scores.allocate();
    Tensor masked_scores({T, T}, DType::Float16); masked_scores.allocate();
    Tensor probs({T, T}, DType::Float16); probs.allocate();
    Tensor ctx_seq({T, HeadDim}, DType::Float16); ctx_seq.allocate();

    for (int64_t qh = 0; qh < NumQHeads; ++qh) {
        const int64_t kvh = qh / QPerKV;
        copy_head_to_seq(q_rope, qh, NumQHeads, q_seq, stream);
        copy_head_to_seq(k_rope, kvh, NumKVHeads, k_seq, stream);
        copy_col_block(v_full, kvh * HeadDim, v_seq, stream);
        matmul_b_transposed(q_seq, k_seq, scores, stream);
        mul(scores, scale, scaled_scores, stream);
        add(scaled_scores, causal_mask, masked_scores, stream);
        softmax_last_dim(masked_scores, probs, stream);
        matmul(probs, v_seq, ctx_seq, stream);
        copy_seq_to_head_block(ctx_seq, attn_out, qh * HeadDim, stream);
    }
    // Single sync after all heads complete
    check_acl(aclrtSynchronizeStream(stream), "attention loop complete");

    Tensor attn_proj({T, Hidden}, DType::Float16); attn_proj.allocate();
    if (use_gated_attn) {
        Tensor gate_sig({T, QMainDim}, DType::Float16); gate_sig.allocate();
        sigmoid(q_gate, gate_sig, stream);
        Tensor attn_gated({T, QMainDim}, DType::Float16); attn_gated.allocate();
        mul(attn_out, gate_sig, attn_gated, stream);
        matmul_b_transposed(attn_gated, *weights.o_proj_weight, attn_proj, stream);
    } else {
        // No gating: project attn_out directly
        matmul_b_transposed(attn_out, *weights.o_proj_weight, attn_proj, stream);
    }

    Tensor after_attn({T, Hidden}, DType::Float16); after_attn.allocate();
    add(hidden, attn_proj, after_attn, stream);

    Tensor mlp_in({T, Hidden}, DType::Float16); mlp_in.allocate();
    rms_norm(after_attn, *weights.post_attention_norm_weight, mlp_in, config.rms_epsilon, stream);

    Tensor gate({T, Intermediate}, DType::Float16); gate.allocate();
    Tensor up({T, Intermediate}, DType::Float16); up.allocate();
    Tensor gate_act({T, Intermediate}, DType::Float16); gate_act.allocate();
    Tensor gated({T, Intermediate}, DType::Float16); gated.allocate();
    Tensor mlp_out({T, Hidden}, DType::Float16); mlp_out.allocate();

    matmul_b_transposed(mlp_in, *weights.gate_proj_weight, gate, stream);
    matmul_b_transposed(mlp_in, *weights.up_proj_weight, up, stream);
    silu_mul(gate, up, gated, stream);
    matmul_b_transposed(gated, *weights.down_proj_weight, mlp_out, stream);
    add(after_attn, mlp_out, out, stream);
}

}  // namespace

void full_attention_decoder_layer(const Tensor& hidden,
                                  const FullAttentionDecoderLayerWeights& weights,
                                  const Tensor& cos_table,
                                  const Tensor& sin_table,
                                  const std::vector<int32_t>& row_to_t,
                                  const FullAttentionDecoderLayerConfig& config,
                                  Tensor& out,
                                  aclrtStream stream) {
    run_full_attention_core(hidden, weights, cos_table, sin_table, row_to_t, config,
                            nullptr, 0, out, stream);
}

void full_attention_decoder_layer_with_cache(const Tensor& hidden,
                                             const FullAttentionDecoderLayerWeights& weights,
                                             const Tensor& cos_table,
                                             const Tensor& sin_table,
                                             const std::vector<int32_t>& row_to_t,
                                             const FullAttentionDecoderLayerConfig& config,
                                             FullAttentionLayerCache& cache,
                                             Tensor& out,
                                             aclrtStream stream) {
    run_full_attention_core(hidden, weights, cos_table, sin_table, row_to_t, config,
                            &cache, 0, out, stream);
}

void linear_attention_decoder_layer_stub(const Tensor& hidden,
                                         const LinearAttentionDecoderLayerStubWeights& weights,
                                         const LinearAttentionDecoderLayerConfig& config,
                                         Tensor& out,
                                         aclrtStream stream) {
    check_ptr(weights.input_norm_weight, "linear input_norm_weight");
    check_ptr(weights.post_attention_norm_weight, "linear post_attention_norm_weight");
    check_ptr(weights.gate_proj_weight, "linear gate_proj_weight");
    check_ptr(weights.up_proj_weight, "linear up_proj_weight");
    check_ptr(weights.down_proj_weight, "linear down_proj_weight");

    if (hidden.shape().size() != 2 || out.shape() != hidden.shape()) {
        throw std::runtime_error("linear decoder layer hidden/out must be [T, H] and same shape");
    }
    if (hidden.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("linear decoder layer requires fp16 hidden/out");
    }

    const int64_t T = hidden.shape()[0];
    const int64_t Hidden = hidden.shape()[1];
    const int64_t Intermediate = [&]{
        const auto& s = weights.gate_proj_weight->shape();
        if (s.size() == 2) {
            if (s[1] == Hidden) return s[0];
            if (s[0] == Hidden) return s[1];
        }
        return s[0];
    }();
    if (weights.input_norm_weight->shape() != std::vector<int64_t>{Hidden} ||
        weights.post_attention_norm_weight->shape() != std::vector<int64_t>{Hidden} ||
        !matmul_shape_ok(weights.gate_proj_weight, Intermediate, Hidden) ||
        !matmul_shape_ok(weights.up_proj_weight, Intermediate, Hidden) ||
        !matmul_shape_ok(weights.down_proj_weight, Hidden, Intermediate)) {
        throw std::runtime_error("linear decoder layer weight shape mismatch");
    }

    Tensor normed({T, Hidden}, DType::Float16); normed.allocate();
    rms_norm(hidden, *weights.input_norm_weight, normed, config.rms_epsilon, stream);

    Tensor after_attn({T, Hidden}, DType::Float16); after_attn.allocate();
    add(hidden, normed, after_attn, stream);

    Tensor mlp_in({T, Hidden}, DType::Float16); mlp_in.allocate();
    rms_norm(after_attn, *weights.post_attention_norm_weight, mlp_in, config.rms_epsilon, stream);

    Tensor gate({T, Intermediate}, DType::Float16); gate.allocate();
    Tensor up({T, Intermediate}, DType::Float16); up.allocate();
    Tensor gate_act({T, Intermediate}, DType::Float16); gate_act.allocate();
    Tensor gated({T, Intermediate}, DType::Float16); gated.allocate();
    Tensor mlp_out({T, Hidden}, DType::Float16); mlp_out.allocate();

    matmul_b_transposed(mlp_in, *weights.gate_proj_weight, gate, stream);
    matmul_b_transposed(mlp_in, *weights.up_proj_weight, up, stream);
    silu_mul(gate, up, gated, stream);
    matmul_b_transposed(gated, *weights.down_proj_weight, mlp_out, stream);
    add(after_attn, mlp_out, out, stream);
}

namespace {

float h16_to_f32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int32_t e = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; e--; }
            mant &= 0x3ffu;
            bits = sign | (static_cast<uint32_t>(e + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

}  // namespace

void linear_attention_decoder_layer(const Tensor& hidden,
                                    const LinearAttentionDecoderLayerWeights& weights,
                                    const LinearAttentionDecoderLayerConfig& config,
                                    Tensor& out,
                                    aclrtStream stream) {
    check_ptr(weights.input_norm_weight, "linear input_norm_weight");
    check_ptr(weights.post_attention_norm_weight, "linear post_attention_norm_weight");
    check_ptr(weights.in_proj_qkv_weight, "linear in_proj_qkv_weight");
    check_ptr(weights.in_proj_z_weight, "linear in_proj_z_weight");
    check_ptr(weights.in_proj_a_weight, "linear in_proj_a_weight");
    check_ptr(weights.in_proj_b_weight, "linear in_proj_b_weight");
    check_ptr(weights.conv1d_weight, "linear conv1d_weight");
    check_ptr(weights.dt_bias, "linear dt_bias");
    check_ptr(weights.a_log, "linear a_log");
    check_ptr(weights.gated_norm_weight, "linear gated_norm_weight");
    check_ptr(weights.out_proj_weight, "linear out_proj_weight");
    check_ptr(weights.gate_proj_weight, "linear gate_proj_weight");
    check_ptr(weights.up_proj_weight, "linear up_proj_weight");
    check_ptr(weights.down_proj_weight, "linear down_proj_weight");

    if (hidden.shape().size() != 2 || hidden.shape()[1] != 1024) {
        throw std::runtime_error("linear decoder layer hidden must be [T, 1024]");
    }
    if (out.shape() != hidden.shape() || out.dtype() != DType::Float16 || hidden.dtype() != DType::Float16) {
        throw std::runtime_error("linear decoder layer hidden/out must match shape and be fp16");
    }

    const int64_t T = hidden.shape()[0];
    const int64_t Hidden = 1024;
    const int64_t NumHeads = 16;
    const int64_t HeadDim = 128;
    const int64_t KeyDim = NumHeads * HeadDim;
    const int64_t ValueDim = NumHeads * HeadDim;
    const int64_t ConvDim = 2 * KeyDim + ValueDim;
    const int64_t Intermediate = [&]{
        const auto& s = weights.gate_proj_weight->shape();
        if (s.size() == 2) {
            if (s[1] == Hidden) return s[0];
            if (s[0] == Hidden) return s[1];
        }
        return s[0];
    }();

    if (!matmul_shape_ok(weights.in_proj_qkv_weight, ConvDim, Hidden) ||
        !matmul_shape_ok(weights.in_proj_z_weight, ValueDim, Hidden) ||
        !matmul_shape_ok(weights.in_proj_a_weight, NumHeads, Hidden) ||
        !matmul_shape_ok(weights.in_proj_b_weight, NumHeads, Hidden) ||
        weights.conv1d_weight->shape() != std::vector<int64_t>{ConvDim, 1, 4} ||
        weights.dt_bias->shape() != std::vector<int64_t>{NumHeads} ||
        weights.a_log->shape() != std::vector<int64_t>{NumHeads} ||
        weights.gated_norm_weight->shape() != std::vector<int64_t>{HeadDim} ||
        !matmul_shape_ok(weights.out_proj_weight, Hidden, ValueDim) ||
        weights.input_norm_weight->shape() != std::vector<int64_t>{Hidden} ||
        weights.post_attention_norm_weight->shape() != std::vector<int64_t>{Hidden} ||
        !matmul_shape_ok(weights.gate_proj_weight, Intermediate, Hidden) ||
        !matmul_shape_ok(weights.up_proj_weight, Intermediate, Hidden) ||
        !matmul_shape_ok(weights.down_proj_weight, Hidden, Intermediate)) {
        throw std::runtime_error("linear decoder layer weight shape mismatch");
    }

    Tensor normed({T, Hidden}, DType::Float16); normed.allocate();
    rms_norm(hidden, *weights.input_norm_weight, normed, config.rms_epsilon, stream);

    Tensor qkv({T, ConvDim}, DType::Float16); qkv.allocate();
    Tensor z({T, ValueDim}, DType::Float16); z.allocate();
    Tensor a({T, NumHeads}, DType::Float16); a.allocate();
    Tensor b({T, NumHeads}, DType::Float16); b.allocate();
    matmul_b_transposed(normed, *weights.in_proj_qkv_weight, qkv, stream);
    matmul_b_transposed(normed, *weights.in_proj_z_weight, z, stream);
    matmul_b_transposed(normed, *weights.in_proj_a_weight, a, stream);
    matmul_b_transposed(normed, *weights.in_proj_b_weight, b, stream);

    Tensor conv({T, ConvDim}, DType::Float16); conv.allocate();
    Tensor mixed({T, ConvDim}, DType::Float16); mixed.allocate();
    linear_causal_conv(qkv, *weights.conv1d_weight, conv, stream);
    silu(conv, mixed, stream);

    std::vector<uint16_t> a_host(static_cast<size_t>(T) * NumHeads);
    std::vector<uint16_t> b_host(static_cast<size_t>(T) * NumHeads);
    std::vector<uint16_t> dt_host(NumHeads);
    std::vector<uint16_t> a_log_host(NumHeads);
    a.copy_to_host(a_host.data(), a_host.size() * sizeof(uint16_t));
    b.copy_to_host(b_host.data(), b_host.size() * sizeof(uint16_t));
    weights.dt_bias->copy_to_host(dt_host.data(), dt_host.size() * sizeof(uint16_t));
    weights.a_log->copy_to_host(a_log_host.data(), a_log_host.size() * sizeof(uint16_t));

    std::vector<uint16_t> beta_h(static_cast<size_t>(T) * NumHeads);
    std::vector<uint16_t> decay_h(static_cast<size_t>(T) * NumHeads);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < NumHeads; ++h) {
            float bv = h16_to_f32(b_host[t * NumHeads + h]);
            float av = h16_to_f32(a_host[t * NumHeads + h]);
            float dtv = h16_to_f32(dt_host[h]);
            float alv = h16_to_f32(a_log_host[h]);
            float g = -std::exp(alv) * softplus(av + dtv);
            beta_h[t * NumHeads + h] = f32_to_f16_bits(sigmoid(bv));
            decay_h[t * NumHeads + h] = f32_to_f16_bits(std::exp(g));
        }
    }

    Tensor beta_dev({T, NumHeads}, DType::Float16);
    Tensor decay_dev({T, NumHeads}, DType::Float16);
    beta_dev.copy_from_host(beta_h.data(), beta_h.size() * sizeof(uint16_t));
    decay_dev.copy_from_host(decay_h.data(), decay_h.size() * sizeof(uint16_t));

    Tensor core_dev({T, ValueDim}, DType::Float16); core_dev.allocate();
    Tensor scratch({136192}, DType::Float32); scratch.allocate();
    linear_gated_delta_rule(mixed, beta_dev, decay_dev, scratch, core_dev, stream);

    Tensor z_silu({T, ValueDim}, DType::Float16); z_silu.allocate();
    silu(z, z_silu, stream);

    Tensor gated({T, ValueDim}, DType::Float16); gated.allocate();
    gated_rms_norm_z(core_dev, z_silu, *weights.gated_norm_weight, gated, stream);

    Tensor attn_proj({T, Hidden}, DType::Float16); attn_proj.allocate();
    matmul_b_transposed(gated, *weights.out_proj_weight, attn_proj, stream);

    Tensor after_attn({T, Hidden}, DType::Float16); after_attn.allocate();
    add(hidden, attn_proj, after_attn, stream);

    Tensor mlp_in({T, Hidden}, DType::Float16); mlp_in.allocate();
    rms_norm(after_attn, *weights.post_attention_norm_weight, mlp_in, config.rms_epsilon, stream);

    Tensor gate({T, Intermediate}, DType::Float16); gate.allocate();
    Tensor up({T, Intermediate}, DType::Float16); up.allocate();
    Tensor gate_act({T, Intermediate}, DType::Float16); gate_act.allocate();
    Tensor gated_mlp({T, Intermediate}, DType::Float16); gated_mlp.allocate();
    Tensor mlp_out({T, Hidden}, DType::Float16); mlp_out.allocate();

    matmul_b_transposed(mlp_in, *weights.gate_proj_weight, gate, stream);
    matmul_b_transposed(mlp_in, *weights.up_proj_weight, up, stream);
    silu_mul(gate, up, gated_mlp, stream);
    matmul_b_transposed(gated_mlp, *weights.down_proj_weight, mlp_out, stream);
    add(after_attn, mlp_out, out, stream);
}

void linear_attention_decoder_layer_with_cache(const Tensor& hidden,
                                               const LinearAttentionDecoderLayerWeights& weights,
                                               const LinearAttentionDecoderLayerConfig& config,
                                               LinearAttentionLayerCache& cache,
                                               Tensor& out,
                                               aclrtStream stream) {
    check_ptr(weights.input_norm_weight, "linear input_norm_weight");
    check_ptr(weights.post_attention_norm_weight, "linear post_attention_norm_weight");
    check_ptr(weights.in_proj_qkv_weight, "linear in_proj_qkv_weight");
    check_ptr(weights.in_proj_z_weight, "linear in_proj_z_weight");
    check_ptr(weights.in_proj_a_weight, "linear in_proj_a_weight");
    check_ptr(weights.in_proj_b_weight, "linear in_proj_b_weight");
    check_ptr(weights.conv1d_weight, "linear conv1d_weight");
    check_ptr(weights.dt_bias, "linear dt_bias");
    check_ptr(weights.a_log, "linear a_log");
    check_ptr(weights.gated_norm_weight, "linear gated_norm_weight");
    check_ptr(weights.out_proj_weight, "linear out_proj_weight");
    check_ptr(weights.gate_proj_weight, "linear gate_proj_weight");
    check_ptr(weights.up_proj_weight, "linear up_proj_weight");
    check_ptr(weights.down_proj_weight, "linear down_proj_weight");

    if (hidden.shape().size() != 2 || hidden.shape()[1] != 1024) {
        throw std::runtime_error("linear decoder layer with_cache hidden must be [T, 1024]");
    }
    if (out.shape() != hidden.shape() || out.dtype() != DType::Float16 || hidden.dtype() != DType::Float16) {
        throw std::runtime_error("linear decoder layer with_cache hidden/out must match shape and be fp16");
    }
    if (cache.conv_buf.shape() != std::vector<int64_t>{3, 6144} ||
        cache.recurrent_state.shape() != std::vector<int64_t>{16, 128, 128}) {
        throw std::runtime_error("linear decoder layer with_cache cache shape mismatch");
    }

    const int64_t T = hidden.shape()[0];
    const int64_t Hidden = 1024;
    const int64_t NumHeads = 16;
    const int64_t HeadDim = 128;
    const int64_t KeyDim = NumHeads * HeadDim;
    const int64_t ValueDim = NumHeads * HeadDim;
    const int64_t ConvDim = 2 * KeyDim + ValueDim;
    const int64_t Intermediate = [&]{
        const auto& s = weights.gate_proj_weight->shape();
        if (s.size() == 2) {
            if (s[1] == Hidden) return s[0];
            if (s[0] == Hidden) return s[1];
        }
        return s[0];
    }();

    Tensor normed({T, Hidden}, DType::Float16); normed.allocate();
    rms_norm(hidden, *weights.input_norm_weight, normed, config.rms_epsilon, stream);

    Tensor qkv({T, ConvDim}, DType::Float16); qkv.allocate();
    Tensor z({T, ValueDim}, DType::Float16); z.allocate();
    Tensor a({T, NumHeads}, DType::Float16); a.allocate();
    Tensor b({T, NumHeads}, DType::Float16); b.allocate();
    matmul_b_transposed(normed, *weights.in_proj_qkv_weight, qkv, stream);
    matmul_b_transposed(normed, *weights.in_proj_z_weight, z, stream);
    matmul_b_transposed(normed, *weights.in_proj_a_weight, a, stream);
    matmul_b_transposed(normed, *weights.in_proj_b_weight, b, stream);

    Tensor conv({T, ConvDim}, DType::Float16); conv.allocate();
    Tensor mixed({T, ConvDim}, DType::Float16); mixed.allocate();
    linear_causal_conv(qkv, *weights.conv1d_weight, conv, stream);
    silu(conv, mixed, stream);

    std::vector<uint16_t> a_host(static_cast<size_t>(T) * NumHeads);
    std::vector<uint16_t> b_host(static_cast<size_t>(T) * NumHeads);
    std::vector<uint16_t> dt_host(NumHeads);
    std::vector<uint16_t> a_log_host(NumHeads);
    a.copy_to_host(a_host.data(), a_host.size() * sizeof(uint16_t));
    b.copy_to_host(b_host.data(), b_host.size() * sizeof(uint16_t));
    weights.dt_bias->copy_to_host(dt_host.data(), dt_host.size() * sizeof(uint16_t));
    weights.a_log->copy_to_host(a_log_host.data(), a_log_host.size() * sizeof(uint16_t));

    std::vector<uint16_t> beta_h(static_cast<size_t>(T) * NumHeads);
    std::vector<uint16_t> decay_h(static_cast<size_t>(T) * NumHeads);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < NumHeads; ++h) {
            float bv = h16_to_f32(b_host[t * NumHeads + h]);
            float av = h16_to_f32(a_host[t * NumHeads + h]);
            float dtv = h16_to_f32(dt_host[h]);
            float alv = h16_to_f32(a_log_host[h]);
            float g = -std::exp(alv) * softplus(av + dtv);
            beta_h[t * NumHeads + h] = f32_to_f16_bits(sigmoid(bv));
            decay_h[t * NumHeads + h] = f32_to_f16_bits(std::exp(g));
        }
    }

    Tensor beta_dev({T, NumHeads}, DType::Float16);
    Tensor decay_dev({T, NumHeads}, DType::Float16);
    beta_dev.copy_from_host(beta_h.data(), beta_h.size() * sizeof(uint16_t));
    decay_dev.copy_from_host(decay_h.data(), decay_h.size() * sizeof(uint16_t));

    // Recurrence over T tokens, advancing cache.recurrent_state. Reuse per-step
    // row tensors to avoid T allocations.
    Tensor core_dev({T, ValueDim}, DType::Float16); core_dev.allocate();
    Tensor mixed_row({1, ConvDim}, DType::Float16); mixed_row.allocate();
    Tensor beta_row({1, NumHeads}, DType::Float16); beta_row.allocate();
    Tensor decay_row({1, NumHeads}, DType::Float16); decay_row.allocate();
    Tensor core_row({1, ValueDim}, DType::Float16); core_row.allocate();
    Tensor step_scratch({8 * 6 * 128}, DType::Float32); step_scratch.allocate();
    for (int64_t t = 0; t < T; ++t) {
        copy_matrix_rows(mixed, t, mixed_row, 0, 1, stream);
        copy_matrix_rows(beta_dev, t, beta_row, 0, 1, stream);
        copy_matrix_rows(decay_dev, t, decay_row, 0, 1, stream);
        linear_gated_delta_rule_step(mixed_row, beta_row, decay_row,
                                     cache.recurrent_state, step_scratch, core_row, stream);
        copy_matrix_rows(core_row, 0, core_dev, t, 1, stream);
    }

    Tensor z_silu({T, ValueDim}, DType::Float16); z_silu.allocate();
    silu(z, z_silu, stream);

    Tensor gated({T, ValueDim}, DType::Float16); gated.allocate();
    gated_rms_norm_z(core_dev, z_silu, *weights.gated_norm_weight, gated, stream);

    Tensor attn_proj({T, Hidden}, DType::Float16); attn_proj.allocate();
    matmul_b_transposed(gated, *weights.out_proj_weight, attn_proj, stream);

    Tensor after_attn({T, Hidden}, DType::Float16); after_attn.allocate();
    add(hidden, attn_proj, after_attn, stream);

    Tensor mlp_in({T, Hidden}, DType::Float16); mlp_in.allocate();
    rms_norm(after_attn, *weights.post_attention_norm_weight, mlp_in, config.rms_epsilon, stream);

    Tensor gate({T, Intermediate}, DType::Float16); gate.allocate();
    Tensor up({T, Intermediate}, DType::Float16); up.allocate();
    Tensor gate_act({T, Intermediate}, DType::Float16); gate_act.allocate();
    Tensor gated_mlp({T, Intermediate}, DType::Float16); gated_mlp.allocate();
    Tensor mlp_out({T, Hidden}, DType::Float16); mlp_out.allocate();

    matmul_b_transposed(mlp_in, *weights.gate_proj_weight, gate, stream);
    matmul_b_transposed(mlp_in, *weights.up_proj_weight, up, stream);
    silu_mul(gate, up, gated_mlp, stream);
    matmul_b_transposed(gated_mlp, *weights.down_proj_weight, mlp_out, stream);
    add(after_attn, mlp_out, out, stream);

    // Update conv buffer with last min(T, 3) pre-conv qkv rows. With the
    // documented "cache must start zero" precondition, leaving the first
    // (3 - min(T,3)) rows untouched preserves the conv1d's implicit-zero-history
    // semantics for the next decode step.
    const int64_t copy_count = std::min<int64_t>(T, 3);
    if (copy_count > 0) {
        copy_matrix_rows(qkv, T - copy_count, cache.conv_buf, 3 - copy_count, copy_count, stream);
    }
}

DecodeState make_decode_state(int64_t max_seq_len,
                              const std::vector<std::string>& layer_types,
                              const FullAttentionDecoderLayerConfig& full_config,
                              aclrtStream stream) {
    if (max_seq_len <= 0) {
        throw std::runtime_error("decode state max_seq_len must be positive");
    }
    DecodeState state;
    state.max_seq_len = max_seq_len;
    state.seq_len = 0;
    const int64_t kv_dim = full_config.num_kv_heads * full_config.head_dim;
    for (const auto& type : layer_types) {
        if (type == "full_attention") {
            FullAttentionLayerCache cache;
            // Layout: [num_kv_heads, max_seq_len, head_dim] for contiguous per-head
            // gather. Old layout [max_seq_len, kv_dim] required context-many memcpy
            // calls per head; new layout needs one contiguous copy per head.
            cache.k_cache = Tensor({full_config.num_kv_heads, max_seq_len, full_config.head_dim}, DType::Float16);
            cache.v_cache = Tensor({full_config.num_kv_heads, max_seq_len, full_config.head_dim}, DType::Float16);
            cache.k_cache.allocate();
            cache.v_cache.allocate();
            check_acl(aclrtMemsetAsync(cache.k_cache.data(), cache.k_cache.size_bytes(), 0,
                                       cache.k_cache.size_bytes(), stream), "memset full k cache");
            check_acl(aclrtMemsetAsync(cache.v_cache.data(), cache.v_cache.size_bytes(), 0,
                                       cache.v_cache.size_bytes(), stream), "memset full v cache");
            state.full.push_back(std::move(cache));
        } else if (type == "linear_attention") {
            LinearAttentionLayerCache cache;
            cache.conv_buf = Tensor({3, 6144}, DType::Float16);
            cache.recurrent_state = Tensor({16, 128, 128}, DType::Float32);
            cache.conv_buf.allocate();
            cache.recurrent_state.allocate();
            check_acl(aclrtMemsetAsync(cache.conv_buf.data(), cache.conv_buf.size_bytes(), 0,
                                       cache.conv_buf.size_bytes(), stream), "memset linear conv cache");
            check_acl(aclrtMemsetAsync(cache.recurrent_state.data(), cache.recurrent_state.size_bytes(), 0,
                                       cache.recurrent_state.size_bytes(), stream), "memset linear recurrent cache");
            state.linear.push_back(std::move(cache));
        } else {
            throw std::runtime_error("unknown layer type for decode state: " + type);
        }
    }
    check_acl(aclrtSynchronizeStream(stream), "make_decode_state sync");
    return state;
}

void full_attention_decoder_layer_step(const Tensor& hidden,
                                       const FullAttentionDecoderLayerWeights& weights,
                                       const Tensor& cos_table,
                                       const Tensor& sin_table,
                                       int32_t pos,
                                       int64_t cache_len,
                                       const FullAttentionDecoderLayerConfig& config,
                                       FullAttentionLayerCache& cache,
                                       Tensor& out,
                                       aclrtStream stream) {
    validate_shapes(hidden, weights, config, out);
    if (hidden.shape()[0] != 1) {
        throw std::runtime_error("full_attention_decoder_layer_step hidden must be [1, H]");
    }
    if (cache_len < 0 || cache_len >= cache.k_cache.shape()[1]) {
        throw std::runtime_error("full_attention_decoder_layer_step cache_len out of range");
    }

    const int64_t Hidden = hidden.shape()[1];
    const int64_t NumQHeads = config.num_q_heads;
    const int64_t NumKVHeads = config.num_kv_heads;
    const int64_t QPerKV = NumQHeads / NumKVHeads;
    const int64_t HeadDim = config.head_dim;
    const int64_t QMainDim = NumQHeads * HeadDim;
    const int64_t KVDim = NumKVHeads * HeadDim;

    // Detect if the model uses gated attention by checking q_proj shape
    const auto& q_shape = weights.q_proj_weight->shape();
    int64_t q_proj_out = 0;
    if (q_shape.size() == 2) {
        if (q_shape[1] == Hidden) q_proj_out = q_shape[0];
        else if (q_shape[0] == Hidden) q_proj_out = q_shape[1];
    }
    const bool use_gated_attn = (q_proj_out == QMainDim * 2);
    const int64_t QProjOut = use_gated_attn ? (QMainDim * 2) : QMainDim;
    const int64_t Intermediate = [&]{
        const auto& s = weights.gate_proj_weight->shape();
        if (s.size() == 2) {
            if (s[1] == Hidden) return s[0];
            if (s[0] == Hidden) return s[1];
        }
        return s[0];
    }();
    const int64_t Context = cache_len + 1;

    // Cache is now [num_kv_heads, max_seq, head_dim]
    if (cache.k_cache.shape().size() != 3 ||
        cache.k_cache.shape()[0] != NumKVHeads ||
        cache.k_cache.shape()[2] != HeadDim ||
        cache.v_cache.shape() != cache.k_cache.shape()) {
        throw std::runtime_error("full_attention_decoder_layer_step cache shape mismatch");
    }

    FullAttentionStepScratch& s = cache.scratch;
    if (!s.ready) {
        // Allocate every scratch buffer once, up front, while it's still
        // early (close to model load time) rather than piecemeal on the
        // first decode step. Reused identically on every subsequent step.
        s.normed = Tensor({1, Hidden}, DType::Float16); s.normed.allocate();
        s.q_full = Tensor({1, QProjOut}, DType::Float16); s.q_full.allocate();
        s.k_full = Tensor({1, KVDim}, DType::Float16); s.k_full.allocate();
        s.v_full = Tensor({1, KVDim}, DType::Float16); s.v_full.allocate();
        s.normed_i8 = Tensor({1, Hidden}, DType::Int8); s.normed_i8.allocate();
        s.normed_scale = Tensor({1}, DType::Float16); s.normed_scale.allocate();
        s.q_only = Tensor({1, QMainDim}, DType::Float16); s.q_only.allocate();
        if (use_gated_attn) {
            s.q_gate = Tensor({1, QMainDim}, DType::Float16); s.q_gate.allocate();
            s.gate_sig = Tensor({1, QMainDim}, DType::Float16); s.gate_sig.allocate();
            s.attn_gated = Tensor({1, QMainDim}, DType::Float16); s.attn_gated.allocate();
        }
        s.q_heads = Tensor({NumQHeads, HeadDim}, DType::Float16); s.q_heads.allocate();
        s.k_heads = Tensor({NumKVHeads, HeadDim}, DType::Float16); s.k_heads.allocate();
        s.q_normed = Tensor({NumQHeads, HeadDim}, DType::Float16); s.q_normed.allocate();
        s.k_normed = Tensor({NumKVHeads, HeadDim}, DType::Float16); s.k_normed.allocate();
        s.q_rope = Tensor({NumQHeads, HeadDim}, DType::Float16); s.q_rope.allocate();
        s.k_rope = Tensor({NumKVHeads, HeadDim}, DType::Float16); s.k_rope.allocate();
        s.attn_out = Tensor({1, QMainDim}, DType::Float16); s.attn_out.allocate();
        s.attn_proj = Tensor({1, Hidden}, DType::Float16); s.attn_proj.allocate();
        s.after_attn = Tensor({1, Hidden}, DType::Float16); s.after_attn.allocate();
        s.mlp_in = Tensor({1, Hidden}, DType::Float16); s.mlp_in.allocate();
        s.gate = Tensor({1, Intermediate}, DType::Float16); s.gate.allocate();
        s.up = Tensor({1, Intermediate}, DType::Float16); s.up.allocate();
        s.gated = Tensor({1, Intermediate}, DType::Float16); s.gated.allocate();
        s.mlp_out = Tensor({1, Hidden}, DType::Float16); s.mlp_out.allocate();
        s.mlp_i8 = Tensor({1, Hidden}, DType::Int8); s.mlp_i8.allocate();
        s.mlp_scale = Tensor({1}, DType::Float16); s.mlp_scale.allocate();
        s.ready = true;
    }

    Tensor& normed = s.normed;
    {
        ProfileScope _p("decode.rms_norm_in", stream);
        rms_norm(hidden, *weights.input_norm_weight, normed, config.rms_epsilon, stream);
    }

    Tensor& q_full = s.q_full;
    Tensor& k_full = s.k_full;
    Tensor& v_full = s.v_full;
    {
        ProfileScope _p("decode.qkv_proj", stream);
        if (w8a8_weight_ready(weights.q_proj_w8)) {
            w8a8_quantize(normed, s.normed_i8, s.normed_scale, stream);
            matmul_decode_w8a8_prequant(s.normed_i8, s.normed_scale, *weights.q_proj_w8, q_full, stream);
            matmul_decode_dispatch(normed, weights.k_proj_weight, weights.k_proj_q, nullptr, k_full, stream);
            matmul_decode_dispatch(normed, weights.v_proj_weight, weights.v_proj_q, nullptr, v_full, stream);
        } else {
            matmul_decode_dispatch(normed, weights.q_proj_weight, weights.q_proj_q, weights.q_proj_w8, q_full, stream);
            matmul_decode_dispatch(normed, weights.k_proj_weight, weights.k_proj_q, weights.k_proj_w8, k_full, stream);
            matmul_decode_dispatch(normed, weights.v_proj_weight, weights.v_proj_q, weights.v_proj_w8, v_full, stream);
        }
    }

    Tensor& q_only = s.q_only;
    if (use_gated_attn) {
        split_q_gate(q_full, NumQHeads, HeadDim, q_only, s.q_gate, stream);
    } else {
        // No gating: q_full is already QMainDim, just copy to q_only
        check_acl(aclrtMemcpyAsync(q_only.data(), q_only.size_bytes(),
                                   q_full.data(), q_full.size_bytes(),
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "q_full -> q_only");
        check_acl(aclrtSynchronizeStream(stream), "q_only copy sync");
    }

    Tensor& q_heads = s.q_heads;
    Tensor& k_heads = s.k_heads;
    copy_heads_from_cols(q_only, NumQHeads, HeadDim, q_heads, stream);
    copy_heads_from_cols(k_full, NumKVHeads, HeadDim, k_heads, stream);

    Tensor& q_normed = s.q_normed;
    Tensor& k_normed = s.k_normed;
    {
        ProfileScope _p("decode.rms_norm_qk", stream);
        rms_norm(q_heads, *weights.q_norm_weight, q_normed, config.rms_epsilon, stream);
        rms_norm(k_heads, *weights.k_norm_weight, k_normed, config.rms_epsilon, stream);
    }

    std::vector<int32_t> q_row_to_t(NumQHeads, pos);
    std::vector<int32_t> k_row_to_t(NumKVHeads, pos);
    Tensor& q_rope = s.q_rope;
    Tensor& k_rope = s.k_rope;
    {
        ProfileScope _p("decode.rope", stream);
        apply_rope_partial(q_normed, cos_table, sin_table, q_row_to_t, config.rotary_dim, q_rope, stream);
        apply_rope_partial(k_normed, cos_table, sin_table, k_row_to_t, config.rotary_dim, k_rope, stream);
    }

    // Write K/V to cache at position cache_len. Cache is [num_kv_heads, max_seq, head_dim].
    // k_rope/v_full are [1, num_kv_heads * head_dim]; scatter each head's slice into its plane.
    const size_t head_bytes = static_cast<size_t>(HeadDim) * 2;  // fp16
    const size_t plane_stride = static_cast<size_t>(cache.k_cache.shape()[1] * HeadDim) * 2;
    for (int64_t h = 0; h < NumKVHeads; ++h) {
        auto* k_dst = static_cast<uint8_t*>(cache.k_cache.data())
                    + h * plane_stride + static_cast<size_t>(cache_len) * head_bytes;
        auto* k_src = static_cast<const uint8_t*>(k_rope.data()) + h * head_bytes;
        check_acl(aclrtMemcpyAsync(k_dst, head_bytes, k_src, head_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream), "k to cache");

        auto* v_dst = static_cast<uint8_t*>(cache.v_cache.data())
                    + h * plane_stride + static_cast<size_t>(cache_len) * head_bytes;
        auto* v_src = static_cast<const uint8_t*>(v_full.data()) + h * head_bytes;
        check_acl(aclrtMemcpyAsync(v_dst, head_bytes, v_src, head_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream), "v to cache");
    }
    check_acl(aclrtSynchronizeStream(stream), "decode k/v cache scatter sync");

    Tensor& attn_out = s.attn_out;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(HeadDim));
    {
        ProfileScope _p("decode.attention", stream);
        incre_flash_attention(q_rope, cache.k_cache, cache.v_cache,
                              Context, NumQHeads, NumKVHeads, HeadDim,
                              attn_scale, attn_out, stream);
    }
    (void)QPerKV;

    Tensor& attn_proj = s.attn_proj;
    {
        ProfileScope _p("decode.o_proj", stream);
        if (use_gated_attn) {
            sigmoid(s.q_gate, s.gate_sig, stream);
            mul(attn_out, s.gate_sig, s.attn_gated, stream);
            matmul_decode_dispatch(s.attn_gated, weights.o_proj_weight, weights.o_proj_q, weights.o_proj_w8, attn_proj, stream);
        } else {
            // No gating: project attn_out directly
            matmul_decode_dispatch(attn_out, weights.o_proj_weight, weights.o_proj_q, weights.o_proj_w8, attn_proj, stream);
        }
    }

    Tensor& after_attn = s.after_attn;
    add(hidden, attn_proj, after_attn, stream);

    Tensor& mlp_in = s.mlp_in;
    {
        ProfileScope _p("decode.rms_norm_post", stream);
        rms_norm(after_attn, *weights.post_attention_norm_weight, mlp_in, config.rms_epsilon, stream);
    }

    Tensor& gate = s.gate;
    Tensor& up = s.up;
    Tensor& gated = s.gated;
    Tensor& mlp_out = s.mlp_out;

    {
        ProfileScope _p("decode.mlp", stream);
        if (w8a8_weight_ready(weights.gate_proj_w8) || w8a8_weight_ready(weights.up_proj_w8)) {
            w8a8_quantize(mlp_in, s.mlp_i8, s.mlp_scale, stream);
            if (w8a8_weight_ready(weights.gate_proj_w8)) {
                matmul_decode_w8a8_prequant(s.mlp_i8, s.mlp_scale, *weights.gate_proj_w8, gate, stream);
            } else {
                matmul_decode_dispatch(mlp_in, weights.gate_proj_weight, weights.gate_proj_q, nullptr, gate, stream);
            }
            if (w8a8_weight_ready(weights.up_proj_w8)) {
                matmul_decode_w8a8_prequant(s.mlp_i8, s.mlp_scale, *weights.up_proj_w8, up, stream);
            } else {
                matmul_decode_dispatch(mlp_in, weights.up_proj_weight, weights.up_proj_q, nullptr, up, stream);
            }
        } else {
            matmul_decode_dispatch(mlp_in, weights.gate_proj_weight, weights.gate_proj_q, weights.gate_proj_w8, gate, stream);
            matmul_decode_dispatch(mlp_in, weights.up_proj_weight, weights.up_proj_q, weights.up_proj_w8, up, stream);
        }
        silu_mul(gate, up, gated, stream);
        matmul_decode_dispatch(gated, weights.down_proj_weight, weights.down_proj_q, weights.down_proj_w8, mlp_out, stream);
    }
    add(after_attn, mlp_out, out, stream);
}

void linear_attention_decoder_layer_step(const Tensor& hidden,
                                         const LinearAttentionDecoderLayerWeights& weights,
                                         const LinearAttentionDecoderLayerConfig& config,
                                         LinearAttentionLayerCache& cache,
                                         Tensor& out,
                                         aclrtStream stream) {
    check_ptr(weights.input_norm_weight, "linear input_norm_weight");
    check_ptr(weights.post_attention_norm_weight, "linear post_attention_norm_weight");
    check_ptr(weights.in_proj_qkv_weight, "linear in_proj_qkv_weight");
    check_ptr(weights.in_proj_z_weight, "linear in_proj_z_weight");
    check_ptr(weights.in_proj_a_weight, "linear in_proj_a_weight");
    check_ptr(weights.in_proj_b_weight, "linear in_proj_b_weight");
    check_ptr(weights.conv1d_weight, "linear conv1d_weight");
    check_ptr(weights.dt_bias, "linear dt_bias");
    check_ptr(weights.a_log, "linear a_log");
    check_ptr(weights.gated_norm_weight, "linear gated_norm_weight");
    check_ptr(weights.out_proj_weight, "linear out_proj_weight");
    check_ptr(weights.gate_proj_weight, "linear gate_proj_weight");
    check_ptr(weights.up_proj_weight, "linear up_proj_weight");
    check_ptr(weights.down_proj_weight, "linear down_proj_weight");

    if (hidden.shape() != std::vector<int64_t>{1, 1024} || out.shape() != hidden.shape() ||
        hidden.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("linear_attention_decoder_layer_step hidden/out must be [1,1024] fp16");
    }
    if (cache.conv_buf.shape() != std::vector<int64_t>{3, 6144} || cache.recurrent_state.shape() != std::vector<int64_t>{16, 128, 128}) {
        throw std::runtime_error("linear_attention_decoder_layer_step cache shape mismatch");
    }

    const int64_t Hidden = 1024;
    const int64_t NumHeads = 16;
    const int64_t HeadDim = 128;
    const int64_t KeyDim = NumHeads * HeadDim;
    const int64_t ValueDim = NumHeads * HeadDim;
    const int64_t ConvDim = 2 * KeyDim + ValueDim;
    const int64_t Intermediate = [&]{
        const auto& s = weights.gate_proj_weight->shape();
        if (s.size() == 2) {
            if (s[1] == Hidden) return s[0];
            if (s[0] == Hidden) return s[1];
        }
        return s[0];
    }();

    LinearAttentionStepScratch& s = cache.scratch;
    if (!s.ready) {
        s.normed = Tensor({1, Hidden}, DType::Float16); s.normed.allocate();
        s.qkv = Tensor({1, ConvDim}, DType::Float16); s.qkv.allocate();
        s.z = Tensor({1, ValueDim}, DType::Float16); s.z.allocate();
        s.a = Tensor({1, NumHeads}, DType::Float16); s.a.allocate();
        s.b = Tensor({1, NumHeads}, DType::Float16); s.b.allocate();
        s.normed_i8 = Tensor({1, Hidden}, DType::Int8); s.normed_i8.allocate();
        s.normed_scale = Tensor({1}, DType::Float16); s.normed_scale.allocate();
        s.conv_input = Tensor({4, ConvDim}, DType::Float16); s.conv_input.allocate();
        s.conv_last = Tensor({1, ConvDim}, DType::Float16); s.conv_last.allocate();
        s.conv_all = Tensor({4, ConvDim}, DType::Float16); s.conv_all.allocate();
        s.mixed = Tensor({1, ConvDim}, DType::Float16); s.mixed.allocate();
        s.beta_dev = Tensor({1, NumHeads}, DType::Float16); s.beta_dev.allocate();
        s.decay_dev = Tensor({1, NumHeads}, DType::Float16); s.decay_dev.allocate();
        s.core_dev = Tensor({1, ValueDim}, DType::Float16); s.core_dev.allocate();
        s.scratch_buf = Tensor({8 * 6 * 128}, DType::Float32); s.scratch_buf.allocate();
        s.z_silu = Tensor({1, ValueDim}, DType::Float16); s.z_silu.allocate();
        s.gated = Tensor({1, ValueDim}, DType::Float16); s.gated.allocate();
        s.attn_proj = Tensor({1, Hidden}, DType::Float16); s.attn_proj.allocate();
        s.after_attn = Tensor({1, Hidden}, DType::Float16); s.after_attn.allocate();
        s.mlp_in = Tensor({1, Hidden}, DType::Float16); s.mlp_in.allocate();
        s.gate = Tensor({1, Intermediate}, DType::Float16); s.gate.allocate();
        s.up = Tensor({1, Intermediate}, DType::Float16); s.up.allocate();
        s.gated_mlp = Tensor({1, Intermediate}, DType::Float16); s.gated_mlp.allocate();
        s.mlp_out = Tensor({1, Hidden}, DType::Float16); s.mlp_out.allocate();
        s.mlp_i8 = Tensor({1, Hidden}, DType::Int8); s.mlp_i8.allocate();
        s.mlp_scale = Tensor({1}, DType::Float16); s.mlp_scale.allocate();
        s.ready = true;
    }

    Tensor& normed = s.normed;
    rms_norm(hidden, *weights.input_norm_weight, normed, config.rms_epsilon, stream);

    Tensor& qkv = s.qkv;
    Tensor& z = s.z;
    Tensor& a = s.a;
    Tensor& b = s.b;
    if (w8a8_weight_ready(weights.in_proj_qkv_w8)) {
        w8a8_quantize(normed, s.normed_i8, s.normed_scale, stream);
        matmul_decode_w8a8_prequant(s.normed_i8, s.normed_scale, *weights.in_proj_qkv_w8, qkv, stream);
        matmul_decode_dispatch(normed, weights.in_proj_z_weight, weights.in_proj_z_q, nullptr, z, stream);
    } else {
        matmul_decode_dispatch(normed, weights.in_proj_qkv_weight, weights.in_proj_qkv_q, weights.in_proj_qkv_w8, qkv, stream);
        matmul_decode_dispatch(normed, weights.in_proj_z_weight, weights.in_proj_z_q, weights.in_proj_z_w8, z, stream);
    }
    matmul_b_transposed(normed, *weights.in_proj_a_weight, a, stream);
    matmul_b_transposed(normed, *weights.in_proj_b_weight, b, stream);

    Tensor& conv_input = s.conv_input;
    copy_matrix_rows(cache.conv_buf, 0, conv_input, 0, 3, stream);
    copy_matrix_rows(qkv, 0, conv_input, 3, 1, stream);
    Tensor& conv_last = s.conv_last;
    if (weights.conv1d_step_weight != nullptr) {
        // Fast path: vectorized kernel computes only the last row directly,
        // skipping the [4, C] generic-conv work whose first 3 rows are unused.
        linear_causal_conv_step(conv_input, *weights.conv1d_step_weight, conv_last, stream);
    } else {
        linear_causal_conv(conv_input, *weights.conv1d_weight, s.conv_all, stream);
        copy_matrix_rows(s.conv_all, 3, conv_last, 0, 1, stream);
    }
    copy_matrix_rows(conv_input, 1, cache.conv_buf, 0, 3, stream);

    Tensor& mixed = s.mixed;
    silu(conv_last, mixed, stream);

    std::vector<uint16_t> a_host(NumHeads);
    std::vector<uint16_t> b_host(NumHeads);
    std::vector<uint16_t> dt_host(NumHeads);
    std::vector<uint16_t> a_log_host(NumHeads);
    a.copy_to_host(a_host.data(), a_host.size() * sizeof(uint16_t));
    b.copy_to_host(b_host.data(), b_host.size() * sizeof(uint16_t));
    weights.dt_bias->copy_to_host(dt_host.data(), dt_host.size() * sizeof(uint16_t));
    weights.a_log->copy_to_host(a_log_host.data(), a_log_host.size() * sizeof(uint16_t));

    std::vector<uint16_t> beta_h(NumHeads);
    std::vector<uint16_t> decay_h(NumHeads);
    for (int64_t h = 0; h < NumHeads; ++h) {
        float bv = h16_to_f32(b_host[h]);
        float av = h16_to_f32(a_host[h]);
        float dtv = h16_to_f32(dt_host[h]);
        float alv = h16_to_f32(a_log_host[h]);
        float g = -std::exp(alv) * softplus(av + dtv);
        beta_h[h] = f32_to_f16_bits(sigmoid(bv));
        decay_h[h] = f32_to_f16_bits(std::exp(g));
    }

    Tensor& beta_dev = s.beta_dev;
    Tensor& decay_dev = s.decay_dev;
    beta_dev.copy_from_host(beta_h.data(), beta_h.size() * sizeof(uint16_t));
    decay_dev.copy_from_host(decay_h.data(), decay_h.size() * sizeof(uint16_t));

    Tensor& core_dev = s.core_dev;
    Tensor& scratch = s.scratch_buf;
    linear_gated_delta_rule_step(mixed, beta_dev, decay_dev, cache.recurrent_state, scratch, core_dev, stream);

    Tensor& z_silu = s.z_silu;
    silu(z, z_silu, stream);

    Tensor& gated = s.gated;
    gated_rms_norm_z(core_dev, z_silu, *weights.gated_norm_weight, gated, stream);

    Tensor& attn_proj = s.attn_proj;
    matmul_decode_dispatch(gated, weights.out_proj_weight, weights.out_proj_q, weights.out_proj_w8, attn_proj, stream);

    Tensor& after_attn = s.after_attn;
    add(hidden, attn_proj, after_attn, stream);

    Tensor& mlp_in = s.mlp_in;
    rms_norm(after_attn, *weights.post_attention_norm_weight, mlp_in, config.rms_epsilon, stream);

    Tensor& gate = s.gate;
    Tensor& up = s.up;
    Tensor& gated_mlp = s.gated_mlp;
    Tensor& mlp_out = s.mlp_out;

    if (w8a8_weight_ready(weights.gate_proj_w8) || w8a8_weight_ready(weights.up_proj_w8)) {
        w8a8_quantize(mlp_in, s.mlp_i8, s.mlp_scale, stream);
        if (w8a8_weight_ready(weights.gate_proj_w8)) {
            matmul_decode_w8a8_prequant(s.mlp_i8, s.mlp_scale, *weights.gate_proj_w8, gate, stream);
        } else {
            matmul_decode_dispatch(mlp_in, weights.gate_proj_weight, weights.gate_proj_q, nullptr, gate, stream);
        }
        if (w8a8_weight_ready(weights.up_proj_w8)) {
            matmul_decode_w8a8_prequant(s.mlp_i8, s.mlp_scale, *weights.up_proj_w8, up, stream);
        } else {
            matmul_decode_dispatch(mlp_in, weights.up_proj_weight, weights.up_proj_q, nullptr, up, stream);
        }
    } else {
        matmul_decode_dispatch(mlp_in, weights.gate_proj_weight, weights.gate_proj_q, weights.gate_proj_w8, gate, stream);
        matmul_decode_dispatch(mlp_in, weights.up_proj_weight, weights.up_proj_q, weights.up_proj_w8, up, stream);
    }
    silu_mul(gate, up, gated_mlp, stream);
    matmul_decode_dispatch(gated_mlp, weights.down_proj_weight, weights.down_proj_q, weights.down_proj_w8, mlp_out, stream);
    add(after_attn, mlp_out, out, stream);
}

}  // namespace minicpmo
