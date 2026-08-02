#include "minicpmo/ops.h"
#include "minicpmo/acl_context.h"
#include "minicpmo/profiling.h"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <aclnnop/aclnn_mm.h>
#include <aclnnop/aclnn_argmax.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_silu.h>
#include <aclnnop/aclnn_sigmoid.h>
#include <aclnnop/aclnn_softmax.h>
#include <aclnnop/aclnn_layer_norm.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_mean.h>
#include <aclnnop/aclnn_rsqrt.h>
#include <aclnnop/aclnn_rms_norm.h>
#include <aclnnop/aclnn_sub.h>
#include <aclnnop/aclnn_convolution.h>
#include <aclnnop/aclnn_gelu_v2.h>
#include <aclnnop/aclnn_batch_matmul.h>
#include <aclnnop/aclnn_permute.h>
#include "aclnn_rms_norm1024_custom.h"
#include "aclnn_linear_causal_conv_custom.h"
#include "aclnn_linear_causal_conv_step_custom.h"
#include "aclnn_matmul_w4a16_custom.h"
#include "aclnn_matmul_w8a8_i32_custom.h"
#include "aclnn_w8a8_quantize_custom.h"
#include "aclnn_w8a8_dequant_custom.h"
#include "aclnn_linear_gated_delta_rule_custom.h"
#include "aclnn_linear_gated_delta_rule_step_custom.h"
#include "aclnn_gated_rms_norm_z_custom.h"
#include "aclnn_attention_step_custom.h"
#include "aclnn_silu_mul_custom.h"
#include "aclnn_matmul_vec_custom.h"
#include "aclnn_matmul_cube_custom.h"

namespace minicpmo {

void embedding_lookup(const Tensor& weight,
                      const std::vector<int32_t>& host_ids,
                      Tensor& out,
                      aclrtStream stream) {
    if (weight.shape().size() != 2) {
        throw std::runtime_error("embedding weight must be 2D [V, H]");
    }
    if (out.dtype() != weight.dtype()) {
        throw std::runtime_error("embedding out dtype must match weight dtype");
    }
    if (out.shape().size() != 2) {
        throw std::runtime_error("embedding out must be 2D [N, H]");
    }

    const int64_t vocab = weight.shape()[0];
    const int64_t hidden = weight.shape()[1];
    const int64_t n = out.shape()[0];

    if (out.shape()[1] != hidden) {
        throw std::runtime_error("embedding out hidden mismatch");
    }
    if (static_cast<int64_t>(host_ids.size()) != n) {
        throw std::runtime_error("embedding host_ids size must equal out rows");
    }

    const size_t row_bytes = static_cast<size_t>(hidden) * dtype_size(weight.dtype());
    auto* weight_base = static_cast<const uint8_t*>(weight.data());
    auto* out_base = static_cast<uint8_t*>(out.data());

    for (int64_t i = 0; i < n; ++i) {
        const int32_t id = host_ids[i];
        if (id < 0 || id >= vocab) {
            throw std::runtime_error("embedding id out of range");
        }
        const void* src = weight_base + static_cast<size_t>(id) * row_bytes;
        void* dst = out_base + static_cast<size_t>(i) * row_bytes;
        check_acl(aclrtMemcpyAsync(dst, row_bytes, src, row_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "aclrtMemcpyAsync embedding row");
    }
    check_acl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream embedding");
}

namespace {

struct AclTensorHandle {
    aclTensor* tensor{nullptr};
    std::vector<int64_t> view_dims;
    std::vector<int64_t> strides;
    std::vector<int64_t> storage_dims;

    ~AclTensorHandle() { if (tensor) aclDestroyTensor(tensor); }
    AclTensorHandle() = default;
    AclTensorHandle(const AclTensorHandle&) = delete;
    AclTensorHandle& operator=(const AclTensorHandle&) = delete;
};

void make_acl_tensor(const Tensor& t, AclTensorHandle& h) {
    h.view_dims = t.shape();
    h.strides.assign(t.shape().size(), 1);
    for (int i = static_cast<int>(t.shape().size()) - 2; i >= 0; --i) {
        h.strides[i] = h.strides[i + 1] * t.shape()[i + 1];
    }
    // An NZ tensor keeps its logical [K, N] view and strides but declares the
    // fractal storage shape, which is how aclnnMm learns it needs no reformat.
    // ACL_FORMAT_FRACTAL_NZ is 29; format 50 (FRACTAL_NZ_C0_16) is accepted by
    // the API but computes wrong results for this layout, so it is not used.
    aclFormat format = ACL_FORMAT_ND;
    if (t.format() == Format::FractalNz && t.shape().size() == 2) {
        h.storage_dims = fractal_nz_storage_dims(t.shape()[0], t.shape()[1]);
        format = ACL_FORMAT_FRACTAL_NZ;
    } else {
        h.storage_dims = t.shape();
    }
    h.tensor = aclCreateTensor(
        h.view_dims.data(), h.view_dims.size(), to_acl_dtype(t.dtype()),
        h.strides.data(), 0, format,
        h.storage_dims.data(), h.storage_dims.size(),
        t.data());
    if (h.tensor == nullptr) {
        throw std::runtime_error("aclCreateTensor returned null");
    }
}

// Build an arbitrary logical view over a tensor's storage: caller supplies the
// view dims and element strides, so a transpose or a reshape costs no copy.
// aclnnBatchMatMul accepts non-contiguous inputs, which is how q @ k^T avoids
// materializing k^T. `storage_elems` is the total element count backing the
// view, used as the declared 1-D storage shape.
void make_acl_view(const Tensor& t,
                   std::vector<int64_t> dims,
                   std::vector<int64_t> strides,
                   int64_t storage_elems,
                   AclTensorHandle& h) {
    h.view_dims = std::move(dims);
    h.strides = std::move(strides);
    h.storage_dims = {storage_elems};
    h.tensor = aclCreateTensor(
        h.view_dims.data(), h.view_dims.size(), to_acl_dtype(t.dtype()),
        h.strides.data(), 0, ACL_FORMAT_ND,
        h.storage_dims.data(), h.storage_dims.size(),
        t.data());
    if (h.tensor == nullptr) {
        throw std::runtime_error("aclCreateTensor returned null (strided view)");
    }
}

}  // namespace

void matmul(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream) {
    if (a.shape().size() != 2 || b.shape().size() != 2 || out.shape().size() != 2) {
        throw std::runtime_error("matmul tensors must be 2D");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("matmul K dim mismatch");
    }
    if (a.shape()[0] != out.shape()[0] || b.shape()[1] != out.shape()[1]) {
        throw std::runtime_error("matmul out shape mismatch");
    }

    AclTensorHandle ha, hb, ho;
    make_acl_tensor(a, ha);
    make_acl_tensor(b, hb);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    constexpr int8_t kCubeMathType = 1;  // ALLOW_FP32_DOWN_PRECISION
    auto ret = aclnnMmGetWorkspaceSize(ha.tensor, hb.tensor, ho.tensor,
                                       kCubeMathType, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnMmGetWorkspaceSize failed: " + std::to_string(ret));
    }

    void* workspace = nullptr;
    if (ws_size > 0) {
        check_acl(aclrtMalloc(&workspace, ws_size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace");
    }

    ret = aclnnMm(workspace, ws_size, executor, stream);
    if (ret != 0) {
        if (workspace) aclrtFree(workspace);
        throw std::runtime_error("aclnnMm failed: " + std::to_string(ret));
    }

    auto sync_ret = aclrtSynchronizeStream(stream);
    if (workspace) aclrtFree(workspace);
    check_acl(sync_ret, "aclrtSynchronizeStream matmul");
}


void matmul_b_transposed(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream) {
    if (a.shape().size() != 2 || b.shape().size() != 2 || out.shape().size() != 2) {
        throw std::runtime_error("matmul_b_transposed tensors must be 2D");
    }
    const int64_t M = a.shape()[0];
    const int64_t K = a.shape()[1];

    // B can be either [N, K] (legacy matmul_b_transposed convention) or [K, N]
    // (pre-transposed for cube fast path). When N != K only one matches.
    // A FractalNz B is always logically [K, N] -- the packing was built that
    // way -- which also resolves the square case that shape alone cannot.
    const bool bIsNz = (b.format() == Format::FractalNz);
    const bool bIsTransposed = !bIsNz && (b.shape()[1] == K);  // storage [N, K]
    const bool bIsNatural    = bIsNz ||
                               ((b.shape()[0] == K) && (b.shape()[1] != K));  // [K, N]
    if (!bIsTransposed && !bIsNatural) {
        throw std::runtime_error("matmul_b_transposed K dim mismatch");
    }
    const int64_t N = bIsTransposed ? b.shape()[0] : b.shape()[1];
    if (a.shape()[0] != out.shape()[0] || N != out.shape()[1]) {
        throw std::runtime_error("matmul_b_transposed out shape mismatch");
    }

    // Custom cube path — DISABLED by default; it is a pessimization in
    // production. It is numerically correct (the old "161001" failure was a
    // bug in the reference-data generator, not the kernel), and it does beat
    // aclnnMm ~2x when the same weights are re-read every call. But that is
    // a cache-hit artifact: decode walks 36 *distinct* layers per token, so
    // every weight read is cold. Measured on the real model, cycling all 36
    // layers' MLP (bench_inproc_layer_cycling):
    //     custom cube : 20.55 ms same-layer -> 51.27 ms cycling  (2.50x)
    //     aclnnMm     : 40.15 ms same-layer -> 42.90 ms cycling  (1.07x)
    // The auto-selected tiling (baseN=256, baseK=64, stepKb=4) reads only
    // 512 B out of each 24 KB row of the [K, N] weight, strided 64x across K,
    // so it collapses on cold data while aclnnMm degrades barely at all.
    // Kept behind MINICPM_USE_CUBE=1 for tiling experiments. It only accepts
    // B in natural [K, N] layout (no on-device transpose), which production
    // weights already have from load_matmul_weight_transposed.
    // Opt-in only: set MINICPM_USE_CUBE=1 to route through the custom cube op.
    // It is OFF by default because it loses on the production access pattern
    // (see the comment block below).
    static const bool cube_enabled = [] {
        const char* e = std::getenv("MINICPM_USE_CUBE");
        return e && e[0] == '1';
    }();
    if (cube_enabled && bIsNatural && a.dtype() == DType::Float16 &&
        b.dtype() == DType::Float16 && out.dtype() == DType::Float16) {
        AclTensorHandle ca, cb, co;
        make_acl_tensor(a, ca);
        make_acl_tensor(b, cb);
        make_acl_tensor(out, co);
        uint64_t cube_ws_size = 0;
        aclOpExecutor* cube_executor = nullptr;
        auto cube_ret = aclnnMatmulCubeCustomGetWorkspaceSize(ca.tensor, cb.tensor, co.tensor,
                                                              &cube_ws_size, &cube_executor);
        if (cube_ret == 0) {
            void* cube_workspace = nullptr;
            if (cube_ws_size > 0) {
                check_acl(aclrtMalloc(&cube_workspace, cube_ws_size, ACL_MEM_MALLOC_HUGE_FIRST),
                          "aclrtMalloc cube workspace");
            }
            auto cube_ret2 = aclnnMatmulCubeCustom(cube_workspace, cube_ws_size, cube_executor, stream);
            if (cube_ret2 == 0) {
                auto cube_sync = aclrtSynchronizeStream(stream);
                if (cube_workspace) aclrtFree(cube_workspace);
                check_acl(cube_sync, "aclrtSynchronizeStream matmul_b_transposed (cube)");
                return;
            }
            if (cube_workspace) aclrtFree(cube_workspace);
            // Launch failed after a successful GetWorkspaceSize -- fall
            // through to the aclnnMm path below rather than throwing.
        }
        // GetWorkspaceSize rejected the shape -- fall through to aclnnMm.
    }

    // CANN's precompiled MatMulV2_FP16 kernel binary doesn't cover the
    // (M >= 64, K > 4096) corner — kernel lookup returns "kernel pointer null"
    // (errno 361001). Empirically M=32 works at any K and M=1024 works at
    // K<=4096. For larger K we tile along the K dim, accumulate partials.
    // Not applicable to a FractalNz B: this slices B along K by byte offset,
    // which assumes row-major storage. NZ interleaves K inside 16x16 blocks,
    // so a contiguous byte range is not a K range.
    constexpr int64_t kKTile = 4096;
    if (!bIsNz && M >= 64 && K > kKTile && a.dtype() == DType::Float16 &&
        b.dtype() == DType::Float16 && out.dtype() == DType::Float16) {
        const int64_t num_chunks = (K + kKTile - 1) / kKTile;
        Tensor accum({M, N}, DType::Float16); accum.allocate();
        check_acl(aclrtMemsetAsync(accum.data(), accum.size_bytes(), 0,
                                   accum.size_bytes(), stream),
                  "matmul K-tile memset");

        const size_t elem = dtype_size(a.dtype());
        for (int64_t i = 0; i < num_chunks; ++i) {
            const int64_t k_start = i * kKTile;
            const int64_t k_chunk = std::min<int64_t>(kKTile, K - k_start);

            Tensor a_chunk(std::vector<int64_t>{M, k_chunk}, a.dtype()); a_chunk.allocate();
            // a[:, k_start : k_start+k_chunk]: copy row by row (M rows, each
            // k_chunk*elem bytes). aclrtMemcpy2dAsync turned out to corrupt
            // the stack on this CANN build; per-row async memcpy is fine.
            for (int64_t r = 0; r < M; ++r) {
                check_acl(aclrtMemcpyAsync(
                    static_cast<uint8_t*>(a_chunk.data()) + r * k_chunk * elem,
                    k_chunk * elem,
                    static_cast<const uint8_t*>(a.data()) + (r * K + k_start) * elem,
                    k_chunk * elem,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                    "matmul K-tile a row slice");
            }

            std::vector<int64_t> b_chunk_shape;
            if (bIsTransposed) b_chunk_shape = {N, k_chunk};
            else               b_chunk_shape = {k_chunk, N};
            Tensor b_chunk(b_chunk_shape, b.dtype()); b_chunk.allocate();

            if (bIsTransposed) {
                // b is [N, K]; want b[:, k_start : k_start+k_chunk] → [N, k_chunk]
                for (int64_t r = 0; r < N; ++r) {
                    check_acl(aclrtMemcpyAsync(
                        static_cast<uint8_t*>(b_chunk.data()) + r * k_chunk * elem,
                        k_chunk * elem,
                        static_cast<const uint8_t*>(b.data()) + (r * K + k_start) * elem,
                        k_chunk * elem,
                        ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                        "matmul K-tile b row slice (transposed)");
                }
            } else {
                // b is [K, N]; want b[k_start : k_start+k_chunk, :] — contiguous block.
                check_acl(aclrtMemcpyAsync(
                    b_chunk.data(), b_chunk.size_bytes(),
                    static_cast<const uint8_t*>(b.data()) + k_start * N * elem,
                    static_cast<size_t>(k_chunk * N) * elem,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                    "matmul K-tile b slice (natural)");
            }

            Tensor partial(std::vector<int64_t>{M, N}, DType::Float16); partial.allocate();
            matmul_b_transposed(a_chunk, b_chunk, partial, stream);
            add(accum, partial, accum, stream);
        }
        check_acl(aclrtMemcpyAsync(out.data(), out.size_bytes(),
                                   accum.data(), accum.size_bytes(),
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "matmul K-tile out copy");
        check_acl(aclrtSynchronizeStream(stream), "matmul K-tile sync");
        return;
    }

    // NOTE ON M<16: aclnnMm requests a workspace exactly the size of B and
    // reformats the whole ND weight into the Cube's fractal tiling on every
    // call below M=16 (101.19 MB for 4096x12288 fp16; 0 at M>=16). Cost is flat
    // from M=1 to M=8 (12.3 ms) then drops to 7.8 ms at M=16, so it is the
    // weight pass, not arithmetic. That reformat is why decode looked
    // memory-bound at ~7 GB/s while plain aclrtMemcpyAsync sustains 22.1 GB/s
    // on the same buffers, and why halving the bytes with int8 made it SLOWER.
    //
    // It is now avoided at the source: weights are packed into FRACTAL_NZ once
    // at load time (see load_matmul_weight_transposed), so aclnnMm gets the
    // layout it wants and the workspace drops to 0.53 MB. Padding M to 16 was
    // also tried and rejected -- 1.58x faster on synthetic weights but 24%
    // SLOWER on the real model (53.52 vs 43.08 ms cycling 36 layers).

    AclTensorHandle ha, hb, ho;
    make_acl_tensor(a, ha);
    make_acl_tensor(out, ho);

    if (bIsNatural) {
        make_acl_tensor(b, hb);  // [K, N], no view dance — aclnnMm handles natural B
    } else {
        // Legacy: build a transposed view for B: storage [N,K], logical [K,N].
        hb.storage_dims = b.shape();
        hb.view_dims = {b.shape()[1], b.shape()[0]};
        hb.strides = {1, b.shape()[1]};
        hb.tensor = aclCreateTensor(
            hb.view_dims.data(), hb.view_dims.size(), to_acl_dtype(b.dtype()),
            hb.strides.data(), 0, ACL_FORMAT_ND,
            hb.storage_dims.data(), hb.storage_dims.size(),
            b.data());
        if (hb.tensor == nullptr) {
            throw std::runtime_error("aclCreateTensor returned null for transposed B view");
        }
    }

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    constexpr int8_t kCubeMathType = 1;
    auto ret = aclnnMmGetWorkspaceSize(ha.tensor, hb.tensor, ho.tensor,
                                       kCubeMathType, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnMmGetWorkspaceSize (B^T) failed: " + std::to_string(ret));
    }
    void* workspace = nullptr;
    if (ws_size > 0) {
        check_acl(aclrtMalloc(&workspace, ws_size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace");
    }
    ret = aclnnMm(workspace, ws_size, executor, stream);
    if (ret != 0) {
        if (workspace) aclrtFree(workspace);
        throw std::runtime_error("aclnnMm (B^T) failed: " + std::to_string(ret));
    }
    auto sync_ret = aclrtSynchronizeStream(stream);
    if (workspace) aclrtFree(workspace);
    check_acl(sync_ret, "aclrtSynchronizeStream matmul_b_transposed");
}

void argmax_last_dim(const Tensor& self, Tensor& out, aclrtStream stream) {
    if (self.shape().empty()) {
        throw std::runtime_error("argmax input must have rank >= 1");
    }
    const int64_t dim = static_cast<int64_t>(self.shape().size() - 1);

    AclTensorHandle hself, hout;
    make_acl_tensor(self, hself);
    make_acl_tensor(out, hout);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnArgMaxGetWorkspaceSize(hself.tensor, dim, false, hout.tensor, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnArgMaxGetWorkspaceSize failed: " + std::to_string(ret));
    }

    void* workspace = nullptr;
    if (ws_size > 0) {
        check_acl(aclrtMalloc(&workspace, ws_size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace");
    }

    ret = aclnnArgMax(workspace, ws_size, executor, stream);
    if (ret != 0) {
        if (workspace) aclrtFree(workspace);
        throw std::runtime_error("aclnnArgMax failed: " + std::to_string(ret));
    }

    auto sync_ret = aclrtSynchronizeStream(stream);
    if (workspace) aclrtFree(workspace);
    check_acl(sync_ret, "aclrtSynchronizeStream argmax");
}

namespace {

void run_op(const char* name,
            uint64_t ws_size,
            aclOpExecutor* executor,
            aclrtStream stream,
            int (*launch)(void*, uint64_t, aclOpExecutor*, aclrtStream)) {
    void* workspace = nullptr;
    if (ws_size > 0) {
        check_acl(aclrtMalloc(&workspace, ws_size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace");
        check_acl(aclrtMemsetAsync(workspace, ws_size, 0, ws_size, stream), "aclrtMemsetAsync workspace");
    }
    auto ret = launch(workspace, ws_size, executor, stream);
    if (ret != 0) {
        if (workspace) aclrtFree(workspace);
        throw std::runtime_error(std::string(name) + " failed: " + std::to_string(ret));
    }
    auto sync_ret = aclrtSynchronizeStream(stream);
    if (workspace) aclrtFree(workspace);
    // Name the op here: an aicore fault surfaces asynchronously at whatever
    // sync comes next, so a bare "aclrtSynchronizeStream failed" says nothing
    // about which kernel actually faulted.
    if (sync_ret != 0) {
        throw std::runtime_error(std::string("sync after ") + name +
                                 " failed with aclError=" + std::to_string(sync_ret));
    }
}

void check_same_shape(const Tensor& a, const Tensor& b, const char* op) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error(std::string(op) + " shape mismatch");
    }
}

}  // namespace

void incre_flash_attention(const Tensor& query,
                           const Tensor& k_cache,
                           const Tensor& v_cache,
                           int64_t context,
                           int64_t num_q_heads,
                           int64_t num_kv_heads,
                           int64_t head_dim,
                           float scale,
                           Tensor& out,
                           aclrtStream stream) {
    if (query.dtype() != DType::Float16 || k_cache.dtype() != DType::Float16 ||
        v_cache.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("incre_flash_attention requires fp16 tensors");
    }
    if (query.numel() != static_cast<size_t>(num_q_heads * head_dim)) {
        throw std::runtime_error("incre_flash_attention query numel mismatch");
    }
    if (out.numel() != static_cast<size_t>(num_q_heads * head_dim)) {
        throw std::runtime_error("incre_flash_attention out numel mismatch");
    }
    // Cache is now [num_kv_heads, max_seq, head_dim]
    if (k_cache.shape().size() != 3 ||
        k_cache.shape()[0] != num_kv_heads ||
        k_cache.shape()[2] != head_dim ||
        v_cache.shape() != k_cache.shape()) {
        throw std::runtime_error("incre_flash_attention cache shape mismatch");
    }
    if (context <= 0 || context > k_cache.shape()[1]) {
        throw std::runtime_error("incre_flash_attention context out of range");
    }

    // Attention is built from BatchMatMul + Muls + Add + Softmax because 310B
    // has no fused attention kernel: IncreFlashAttention and
    // PromptFlashAttention ship kernels only for 310p/910b/910_93 (see
    // opp/built-in/op_impl/ai_core/tbe/config/), and PromptFlashAttention
    // returns 561103 here.
    //
    // KEY PERFORMANCE ISSUE: CANN JIT compiles a new kernel binary for every
    // distinct (M, N, K) shape it sees. Without bucketing, context grows by 1
    // every step (15, 16, 17, ...) so the score shape is fresh on every token,
    // triggering a JIT miss each time — a 30-token generation used to see
    // thousands of compilations, ~17 s/token on a cold cache.
    //
    // Fix: round context up to the nearest power-of-two bucket, so CANN sees
    // one shape per bucket and reuses it for every token in that bucket. The
    // positions between context and bucket are masked to -inf before the
    // softmax, so whatever the cache holds there cannot reach the output.
    static constexpr int64_t kBuckets[] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    const int64_t max_seq = k_cache.shape()[1];
    int64_t bucket = kBuckets[std::size(kBuckets) - 1];
    for (int64_t b : kBuckets) {
        if (b >= context) { bucket = b; break; }
    }
    // The bucket is read straight out of the cache, so it cannot exceed the
    // cache's own sequence dimension. context <= max_seq always, so clamping
    // here only ever trims padding.
    if (bucket > max_seq) bucket = max_seq;

    // Batch the whole attention over the KV-head dimension. GQA makes this
    // exact: 32 query heads over 8 KV heads is [8, 4, head_dim], and the cache
    // is already [num_kv_heads, max_seq, head_dim], so both sides share the
    // same batch axis with no reshuffling.
    //
    // The previous version looped 8 KV heads x 4 query heads and issued ~7 ops
    // per query head. Attention moves only ~7 MB of KV per token but was taking
    // 36% of the token's wall clock: the cost was op launch (~68 us each,
    // ~8300 launches per token), not memory or math. Batching collapses that to
    // 4 launches per layer regardless of head count.
    const int64_t group_size = num_q_heads / num_kv_heads;
    if (group_size * num_kv_heads != num_q_heads) {
        throw std::runtime_error("incre_flash_attention num_q_heads must be a multiple of num_kv_heads");
    }

    // Pad mask [num_kv_heads, group_size, bucket]: 0.0 for real positions,
    // -65504 (fp16 min) for pads, so padded lanes die in the softmax.
    //
    // Materialized at full shape rather than broadcast from a [bucket] row with
    // stride 0: a stride-0 view declares an extent far larger than its backing
    // allocation, which is a sharp edge for a kernel that sizes its reads off
    // the declared extent. One host->device copy per layer is a few hundred KB
    // at most.
    const size_t mask_elems = static_cast<size_t>(num_kv_heads * group_size * bucket);
    std::vector<uint16_t> mask_host(mask_elems, 0xFBFFu);
    for (int64_t b = 0; b < num_kv_heads * group_size; ++b) {
        uint16_t* row = mask_host.data() + static_cast<size_t>(b) * static_cast<size_t>(bucket);
        for (int64_t t = 0; t < context; ++t) row[t] = 0x0000u;
    }
    Tensor pad_mask({num_kv_heads, group_size, bucket}, DType::Float16);
    pad_mask.allocate();
    pad_mask.copy_from_host(mask_host.data(), mask_elems * sizeof(uint16_t));

    Tensor scores({num_kv_heads, group_size, bucket}, DType::Float16); scores.allocate();

    // No gather: BatchMatMul reads the cache planes in place through strided
    // views. The cache is [num_kv_heads, max_seq, head_dim], so a plane is a
    // batch and the leading `bucket` rows of it are the window we want. Rows
    // between context and bucket hold whatever the cache holds there -- zeros
    // from allocation, or stale K/V from an earlier sequence -- and the mask
    // discards them either way, so they never need to be copied or cleared.
    //
    // This removes two bucket-sized buffers, their memsets, and 16 memcpys per
    // layer per token. It also drops peak memory, which matters: the weights
    // already hold 10.4 GB of a 23 GB pool shared with the host.
    const int64_t cache_elems = num_kv_heads * max_seq * head_dim;

    // scores[b, g, :] = (q[b, g, :] . k[b, :, :]^T) * scale + mask
    {
        ProfileScope _p("attn.qk_softmax", stream);
        AclTensorHandle hq, hk, hscores;
        // query is [num_q_heads * head_dim] contiguous, viewed as [B, G, D].
        make_acl_view(query, {num_kv_heads, group_size, head_dim},
                      {group_size * head_dim, head_dim, 1},
                      num_q_heads * head_dim, hq);
        // k_cache [B, max_seq, D] -> [B, D, bucket]: transposed on its last two
        // axes so BatchMatMul consumes it directly, and windowed to `bucket`
        // rows. No k^T is materialized and nothing is copied.
        make_acl_view(k_cache, {num_kv_heads, head_dim, bucket},
                      {max_seq * head_dim, 1, head_dim}, cache_elems, hk);
        make_acl_tensor(scores, hscores);

        constexpr int8_t kCubeMathType = 1;  // ALLOW_FP32_DOWN_PRECISION
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        check_acl(aclnnBatchMatMulGetWorkspaceSize(hq.tensor, hk.tensor, hscores.tensor,
                                                   kCubeMathType, &ws_size, &executor), "qk bmm ws");
        run_op("aclnnBatchMatMul(qk)", ws_size, executor, stream, aclnnBatchMatMul);

        aclScalar* scale_scalar = aclCreateScalar(&scale, ACL_FLOAT);
        ws_size = 0; executor = nullptr;
        check_acl(aclnnMulsGetWorkspaceSize(hscores.tensor, scale_scalar, hscores.tensor,
                                            &ws_size, &executor), "muls ws");
        run_op("aclnnMuls", ws_size, executor, stream, aclnnMuls);
        aclDestroyScalar(scale_scalar);

        AclTensorHandle hmask;
        make_acl_tensor(pad_mask, hmask);
        float alpha_value = 1.0f;
        aclScalar* alpha = aclCreateScalar(&alpha_value, ACL_FLOAT);
        ws_size = 0; executor = nullptr;
        check_acl(aclnnAddGetWorkspaceSize(hscores.tensor, hmask.tensor, alpha, hscores.tensor,
                                           &ws_size, &executor), "mask add ws");
        run_op("aclnnAdd", ws_size, executor, stream, aclnnAdd);
        aclDestroyScalar(alpha);

        softmax_last_dim(scores, scores, stream);
    }

    // out[b, g, :] = scores[b, g, :] @ v[b, :, :]; the [B, G, D] result is
    // exactly out's existing [num_q_heads, head_dim] layout, so it is written
    // in place with no scatter.
    {
        ProfileScope _p("attn.av_matmul", stream);
        AclTensorHandle hscores, hv, hout;
        make_acl_tensor(scores, hscores);
        make_acl_view(v_cache, {num_kv_heads, bucket, head_dim},
                      {max_seq * head_dim, head_dim, 1}, cache_elems, hv);
        make_acl_view(out, {num_kv_heads, group_size, head_dim},
                      {group_size * head_dim, head_dim, 1},
                      num_q_heads * head_dim, hout);

        constexpr int8_t kCubeMathType = 1;
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        check_acl(aclnnBatchMatMulGetWorkspaceSize(hscores.tensor, hv.tensor, hout.tensor,
                                                   kCubeMathType, &ws_size, &executor), "av bmm ws");
        run_op("aclnnBatchMatMul(av)", ws_size, executor, stream, aclnnBatchMatMul);
    }

    check_acl(aclrtSynchronizeStream(stream), "incre_flash_attention sync");
}

void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, aclrtStream stream) {
    if (gate.dtype() != DType::Float16 || up.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("silu_mul requires fp16 tensors");
    }
    if (gate.shape() != up.shape() || gate.shape() != out.shape()) {
        throw std::runtime_error("silu_mul shape mismatch");
    }

    // Decompose into silu(gate) * up since aclnnSiluMulCustom is not supported
    Tensor gate_silu(gate.shape(), DType::Float16);
    gate_silu.allocate();
    silu(gate, gate_silu, stream);
    mul(gate_silu, up, out, stream);
}

void add(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream) {
    check_same_shape(a, b, "add");
    check_same_shape(a, out, "add");
    AclTensorHandle ha, hb, ho;
    make_acl_tensor(a, ha);
    make_acl_tensor(b, hb);
    make_acl_tensor(out, ho);
    float alpha_value = 1.0f;
    aclScalar* alpha = aclCreateScalar(&alpha_value, ACL_FLOAT);
    if (alpha == nullptr) throw std::runtime_error("aclCreateScalar(alpha) failed");
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(ha.tensor, hb.tensor, alpha, ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        aclDestroyScalar(alpha);
        throw std::runtime_error("aclnnAddGetWorkspaceSize failed: " + std::to_string(ret));
    }
    try {
        run_op("aclnnAdd", ws_size, executor, stream, aclnnAdd);
    } catch (...) {
        aclDestroyScalar(alpha);
        throw;
    }
    aclDestroyScalar(alpha);
}

void mul(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream) {
    check_same_shape(a, b, "mul");
    check_same_shape(a, out, "mul");
    AclTensorHandle ha, hb, ho;
    make_acl_tensor(a, ha);
    make_acl_tensor(b, hb);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(ha.tensor, hb.tensor, ho.tensor, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnMulGetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnMul", ws_size, executor, stream, aclnnMul);
}

void silu(const Tensor& self, Tensor& out, aclrtStream stream) {
    check_same_shape(self, out, "silu");
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnSiluGetWorkspaceSize(hs.tensor, ho.tensor, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnSiluGetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnSilu", ws_size, executor, stream, aclnnSilu);
}

void sigmoid(const Tensor& self, Tensor& out, aclrtStream stream) {
    check_same_shape(self, out, "sigmoid");
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnSigmoidGetWorkspaceSize(hs.tensor, ho.tensor, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnSigmoidGetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnSigmoid", ws_size, executor, stream, aclnnSigmoid);
}

void softmax_last_dim(const Tensor& self, Tensor& out, aclrtStream stream) {
    check_same_shape(self, out, "softmax");
    if (self.shape().empty()) throw std::runtime_error("softmax input must have rank >= 1");
    const int64_t dim = static_cast<int64_t>(self.shape().size() - 1);
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnSoftmaxGetWorkspaceSize(hs.tensor, dim, ho.tensor, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnSoftmaxGetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnSoftmax", ws_size, executor, stream, aclnnSoftmax);
}

void rms_norm1024(const Tensor& x, const Tensor& gamma, Tensor& out,
                  double epsilon, aclrtStream stream) {
    if (x.dtype() != DType::Float16 || gamma.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("rms_norm1024 requires fp16 inputs");
    }
    if (x.shape() != out.shape()) {
        throw std::runtime_error("rms_norm1024 x/out shape mismatch");
    }
    if (x.shape().empty() || x.shape().back() != 1024) {
        throw std::runtime_error("rms_norm1024 last dim must be 1024");
    }
    if (gamma.shape().size() != 1 || gamma.shape()[0] != 1024) {
        throw std::runtime_error("rms_norm1024 gamma shape must be [1024]");
    }
    AclTensorHandle hx, hg, ho;
    make_acl_tensor(x, hx);
    make_acl_tensor(gamma, hg);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnRmsNorm1024CustomGetWorkspaceSize(hx.tensor, hg.tensor, epsilon,
                                                     ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnRmsNorm1024CustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnRmsNorm1024Custom", ws_size, executor, stream, aclnnRmsNorm1024Custom);
}

void rms_norm(const Tensor& x, const Tensor& gamma, Tensor& out,
              double epsilon, aclrtStream stream) {
    if (x.shape() != out.shape()) {
        throw std::runtime_error("rms_norm x/out shape mismatch");
    }
    if (x.shape().empty()) {
        throw std::runtime_error("rms_norm input must have rank >= 1");
    }
    const int64_t hidden = x.shape().back();
    if (gamma.shape().size() != 1 || gamma.shape()[0] != hidden) {
        throw std::runtime_error("rms_norm gamma shape must be [H] matching last dim of x");
    }
    if (x.dtype() != gamma.dtype() || x.dtype() != out.dtype()) {
        throw std::runtime_error("rms_norm dtype mismatch (x/gamma/out must match)");
    }

    // aclnnRmsNorm (fused single-kernel path) returns error 161001 on
    // Ascend 310B for all tested shapes — not supported.
    //
    // Manual decomposition with fp32 intermediate variance:
    // hidden states carry "massive activation" outliers (~1e4-2e4 magnitude);
    // squaring those in fp16 (max ~65504) overflows, corrupting the
    // normalization for the whole row. Computing x*x and the mean-reduce in
    // fp32 avoids the overflow. The rstd is cast back to x's dtype (fp16)
    // before the broadcast multiply so the rest of the forward pass stays fp16.
    std::vector<int64_t> reduce_shape = x.shape();
    reduce_shape.back() = 1;

    Tensor x_f32(x.shape(), DType::Float32);  x_f32.allocate();
    Tensor x_sq(x.shape(), DType::Float32);   x_sq.allocate();
    Tensor mean_f32(reduce_shape, DType::Float32); mean_f32.allocate();
    Tensor rstd(reduce_shape, x.dtype());      rstd.allocate();
    Tensor scaled(x.shape(), x.dtype());       scaled.allocate();

    // 0) x_f32 = cast(x, fp32)
    cast(x, x_f32, stream);

    // 1) x_sq = x_f32 * x_f32
    {
        AclTensorHandle hx, hsq;
        make_acl_tensor(x_f32, hx);
        make_acl_tensor(x_sq, hsq);
        uint64_t ws_size = 0; aclOpExecutor* executor = nullptr;
        auto ret = aclnnMulGetWorkspaceSize(hx.tensor, hx.tensor, hsq.tensor, &ws_size, &executor);
        if (ret != 0) throw std::runtime_error("rms_norm Mul(x,x) ws failed: " + std::to_string(ret));
        run_op("rms_norm Mul(x,x)", ws_size, executor, stream, aclnnMul);
    }

    // 2) mean_f32 = mean(x_sq, dim=-1, keepDim=true)
    {
        AclTensorHandle hsq, hmean;
        make_acl_tensor(x_sq, hsq);
        make_acl_tensor(mean_f32, hmean);
        const int64_t last_dim = static_cast<int64_t>(x.shape().size() - 1);
        std::vector<int64_t> dim_data{last_dim};
        aclIntArray* dim = aclCreateIntArray(dim_data.data(), dim_data.size());
        if (dim == nullptr) throw std::runtime_error("rms_norm aclCreateIntArray failed");
        uint64_t ws_size = 0; aclOpExecutor* executor = nullptr;
        auto ret = aclnnMeanGetWorkspaceSize(hsq.tensor, dim, true,
                                             to_acl_dtype(DType::Float32),
                                             hmean.tensor, &ws_size, &executor);
        if (ret != 0) { aclDestroyIntArray(dim); throw std::runtime_error("rms_norm Mean ws failed: " + std::to_string(ret)); }
        try { run_op("rms_norm Mean", ws_size, executor, stream, aclnnMean); }
        catch (...) { aclDestroyIntArray(dim); throw; }
        aclDestroyIntArray(dim);
    }

    // 3a) mean_f32 += eps
    {
        AclTensorHandle hm;
        make_acl_tensor(mean_f32, hm);
        float eps_f = static_cast<float>(epsilon), alpha_f = 1.0f;
        aclScalar* eps_s = aclCreateScalar(&eps_f, ACL_FLOAT);
        aclScalar* alpha_s = aclCreateScalar(&alpha_f, ACL_FLOAT);
        if (!eps_s || !alpha_s) { aclDestroyScalar(eps_s); aclDestroyScalar(alpha_s); throw std::runtime_error("rms_norm aclCreateScalar failed"); }
        uint64_t ws_size = 0; aclOpExecutor* executor = nullptr;
        auto ret = aclnnAddsGetWorkspaceSize(hm.tensor, eps_s, alpha_s, hm.tensor, &ws_size, &executor);
        if (ret != 0) { aclDestroyScalar(eps_s); aclDestroyScalar(alpha_s); throw std::runtime_error("rms_norm Adds ws failed: " + std::to_string(ret)); }
        try { run_op("rms_norm Adds", ws_size, executor, stream, aclnnAdds); }
        catch (...) { aclDestroyScalar(eps_s); aclDestroyScalar(alpha_s); throw; }
        aclDestroyScalar(eps_s); aclDestroyScalar(alpha_s);
    }

    // 3b) mean_f32 = rsqrt(mean_f32); cast to rstd (fp16)
    {
        AclTensorHandle hm;
        make_acl_tensor(mean_f32, hm);
        uint64_t ws_size = 0; aclOpExecutor* executor = nullptr;
        auto ret = aclnnRsqrtGetWorkspaceSize(hm.tensor, hm.tensor, &ws_size, &executor);
        if (ret != 0) throw std::runtime_error("rms_norm Rsqrt ws failed: " + std::to_string(ret));
        run_op("rms_norm Rsqrt", ws_size, executor, stream, aclnnRsqrt);
        cast(mean_f32, rstd, stream);
    }

    // 4) scaled = x * rstd  (broadcast)
    {
        AclTensorHandle hx, hrstd, hout;
        make_acl_tensor(x, hx);
        make_acl_tensor(rstd, hrstd);
        make_acl_tensor(scaled, hout);
        uint64_t ws_size = 0; aclOpExecutor* executor = nullptr;
        auto ret = aclnnMulGetWorkspaceSize(hx.tensor, hrstd.tensor, hout.tensor, &ws_size, &executor);
        if (ret != 0) throw std::runtime_error("rms_norm Mul(x,rstd) ws failed: " + std::to_string(ret));
        run_op("rms_norm Mul(x,rstd)", ws_size, executor, stream, aclnnMul);
    }

    // 5) out = scaled * gamma  (standard Qwen-style RMSNorm, NOT Gemma-style 1+gamma)
    {
        AclTensorHandle hscaled, hg, hout;
        make_acl_tensor(scaled, hscaled);
        make_acl_tensor(gamma, hg);
        make_acl_tensor(out, hout);
        uint64_t ws_size = 0; aclOpExecutor* executor = nullptr;
        auto ret = aclnnMulGetWorkspaceSize(hscaled.tensor, hg.tensor, hout.tensor, &ws_size, &executor);
        if (ret != 0) throw std::runtime_error("rms_norm Mul(*,gamma) ws failed: " + std::to_string(ret));
        run_op("rms_norm Mul(*,gamma)", ws_size, executor, stream, aclnnMul);
    }
}

void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                Tensor& out, double epsilon, aclrtStream stream) {
    check_same_shape(x, out, "layer_norm");
    if (x.shape().empty()) {
        throw std::runtime_error("layer_norm input must have rank >= 1");
    }
    if (weight.shape() != bias.shape() || weight.shape().size() != 1) {
        throw std::runtime_error("layer_norm weight/bias must be 1D and same shape");
    }
    if (weight.shape()[0] != x.shape().back()) {
        throw std::runtime_error("layer_norm hidden mismatch");
    }
    AclTensorHandle hx, hw, hb, ho;
    make_acl_tensor(x, hx);
    make_acl_tensor(weight, hw);
    make_acl_tensor(bias, hb);
    make_acl_tensor(out, ho);

    std::vector<int64_t> normalized = weight.shape();
    aclIntArray* norm_shape = aclCreateIntArray(normalized.data(), normalized.size());
    if (norm_shape == nullptr) {
        throw std::runtime_error("aclCreateIntArray(normalized_shape) failed");
    }

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnLayerNormGetWorkspaceSize(hx.tensor, norm_shape, hw.tensor, hb.tensor,
                                              epsilon, ho.tensor, nullptr, nullptr,
                                              &ws_size, &executor);
    if (ret != 0) {
        aclDestroyIntArray(norm_shape);
        throw std::runtime_error("aclnnLayerNormGetWorkspaceSize failed: " + std::to_string(ret));
    }
    try {
        run_op("aclnnLayerNorm", ws_size, executor, stream, aclnnLayerNorm);
    } catch (...) {
        aclDestroyIntArray(norm_shape);
        throw;
    }
    aclDestroyIntArray(norm_shape);
}

void cast(const Tensor& self, Tensor& out, aclrtStream stream) {
    if (self.shape() != out.shape()) {
        throw std::runtime_error("cast shape mismatch");
    }
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnCastGetWorkspaceSize(hs.tensor, to_acl_dtype(out.dtype()), ho.tensor,
                                         &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnCastGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnCast", ws_size, executor, stream, aclnnCast);
}

namespace {

void d2d_row_copy(const void* src, size_t src_row_stride_bytes,
                  void* dst, size_t dst_row_stride_bytes,
                  size_t row_bytes, int64_t rows, aclrtStream stream) {
    auto* s = static_cast<const uint8_t*>(src);
    auto* d = static_cast<uint8_t*>(dst);
    for (int64_t r = 0; r < rows; ++r) {
        check_acl(aclrtMemcpyAsync(d + r * dst_row_stride_bytes, row_bytes,
                                   s + r * src_row_stride_bytes, row_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "aclrtMemcpyAsync row copy");
    }
}

}  // namespace

void apply_rope_partial(const Tensor& x,
                        const Tensor& cos_table,
                        const Tensor& sin_table,
                        const std::vector<int32_t>& row_to_t,
                        int64_t rot,
                        Tensor& out,
                        aclrtStream stream) {
    if (x.dtype() != DType::Float16 || cos_table.dtype() != DType::Float16 ||
        sin_table.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("apply_rope_partial requires fp16 tensors");
    }
    if (x.shape().size() != 2 || out.shape().size() != 2) {
        throw std::runtime_error("apply_rope_partial expects x/out shape [N, D]");
    }
    if (x.shape() != out.shape()) {
        throw std::runtime_error("apply_rope_partial x/out shape mismatch");
    }
    if (cos_table.shape().size() != 2 || sin_table.shape().size() != 2 ||
        cos_table.shape() != sin_table.shape()) {
        throw std::runtime_error("apply_rope_partial cos/sin must have shape [T, rot/2]");
    }

    const int64_t N = x.shape()[0];
    const int64_t D = x.shape()[1];
    if (rot <= 0 || rot > D || rot % 2 != 0) {
        throw std::runtime_error("apply_rope_partial rot must be even and <= D");
    }
    const int64_t HalfRot = rot / 2;
    if (cos_table.shape()[1] != HalfRot) {
        throw std::runtime_error("apply_rope_partial cos/sin last dim must be rot/2");
    }
    if (static_cast<int64_t>(row_to_t.size()) != N) {
        throw std::runtime_error("apply_rope_partial row_to_t size must equal N");
    }
    const int64_t T = cos_table.shape()[0];
    for (auto t : row_to_t) {
        if (t < 0 || t >= T) {
            throw std::runtime_error("apply_rope_partial row_to_t entry out of range");
        }
    }

    const size_t fp16_size = sizeof(uint16_t);
    const size_t row_bytes_x = static_cast<size_t>(D) * fp16_size;
    const size_t half_bytes = static_cast<size_t>(HalfRot) * fp16_size;
    const size_t tail_bytes = static_cast<size_t>(D - rot) * fp16_size;

    // Allocate slice tensors [N, HalfRot]
    Tensor x1({N, HalfRot}, DType::Float16); x1.allocate();
    Tensor x2({N, HalfRot}, DType::Float16); x2.allocate();
    Tensor cos_e({N, HalfRot}, DType::Float16); cos_e.allocate();
    Tensor sin_e({N, HalfRot}, DType::Float16); sin_e.allocate();
    Tensor a({N, HalfRot}, DType::Float16); a.allocate();
    Tensor b({N, HalfRot}, DType::Float16); b.allocate();
    Tensor y1({N, HalfRot}, DType::Float16); y1.allocate();
    Tensor y2({N, HalfRot}, DType::Float16); y2.allocate();

    auto* x_base = static_cast<const uint8_t*>(x.data());
    auto* out_base = static_cast<uint8_t*>(out.data());
    auto* cos_base = static_cast<const uint8_t*>(cos_table.data());
    auto* sin_base = static_cast<const uint8_t*>(sin_table.data());

    // 1) gather slices: x1, x2, cos_e, sin_e
    for (int64_t n = 0; n < N; ++n) {
        const uint8_t* x_row = x_base + static_cast<size_t>(n) * row_bytes_x;
        check_acl(aclrtMemcpyAsync(static_cast<uint8_t*>(x1.data()) + n * half_bytes, half_bytes,
                                   x_row + 0 * half_bytes, half_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "rope copy x1");
        check_acl(aclrtMemcpyAsync(static_cast<uint8_t*>(x2.data()) + n * half_bytes, half_bytes,
                                   x_row + 1 * half_bytes, half_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "rope copy x2");
        const int64_t t = row_to_t[n];
        const uint8_t* cos_row = cos_base + static_cast<size_t>(t) * half_bytes;
        const uint8_t* sin_row = sin_base + static_cast<size_t>(t) * half_bytes;
        check_acl(aclrtMemcpyAsync(static_cast<uint8_t*>(cos_e.data()) + n * half_bytes, half_bytes,
                                   cos_row, half_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "rope copy cos");
        check_acl(aclrtMemcpyAsync(static_cast<uint8_t*>(sin_e.data()) + n * half_bytes, half_bytes,
                                   sin_row, half_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "rope copy sin");
    }
    check_acl(aclrtSynchronizeStream(stream), "rope gather sync");

    // 2) a = x1 * cos, b = x2 * sin, y1 = a - b
    mul(x1, cos_e, a, stream);
    mul(x2, sin_e, b, stream);
    {
        AclTensorHandle ha, hb, hy1;
        make_acl_tensor(a, ha);
        make_acl_tensor(b, hb);
        make_acl_tensor(y1, hy1);
        float alpha_f = 1.0f;
        aclScalar* alpha = aclCreateScalar(&alpha_f, ACL_FLOAT);
        if (!alpha) throw std::runtime_error("rope sub alpha alloc failed");
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnSubGetWorkspaceSize(ha.tensor, hb.tensor, alpha, hy1.tensor,
                                            &ws_size, &executor);
        if (ret != 0) {
            aclDestroyScalar(alpha);
            throw std::runtime_error("rope Sub ws failed: " + std::to_string(ret));
        }
        try {
            run_op("rope Sub", ws_size, executor, stream, aclnnSub);
        } catch (...) { aclDestroyScalar(alpha); throw; }
        aclDestroyScalar(alpha);
    }

    // 3) a = x2 * cos, b = x1 * sin, y2 = a + b
    mul(x2, cos_e, a, stream);
    mul(x1, sin_e, b, stream);
    add(a, b, y2, stream);

    // 4) scatter y1, y2 to out[:, 0:HalfRot], out[:, HalfRot:rot]; tail copy
    for (int64_t n = 0; n < N; ++n) {
        uint8_t* o_row = out_base + static_cast<size_t>(n) * row_bytes_x;
        check_acl(aclrtMemcpyAsync(o_row + 0 * half_bytes, half_bytes,
                                   static_cast<uint8_t*>(y1.data()) + n * half_bytes, half_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "rope scatter y1");
        check_acl(aclrtMemcpyAsync(o_row + 1 * half_bytes, half_bytes,
                                   static_cast<uint8_t*>(y2.data()) + n * half_bytes, half_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "rope scatter y2");
        if (tail_bytes > 0) {
            const uint8_t* x_row = x_base + static_cast<size_t>(n) * row_bytes_x;
            check_acl(aclrtMemcpyAsync(o_row + static_cast<size_t>(rot) * fp16_size, tail_bytes,
                                       x_row + static_cast<size_t>(rot) * fp16_size, tail_bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                      "rope scatter tail");
        }
    }
    check_acl(aclrtSynchronizeStream(stream), "rope scatter sync");
}

void linear_causal_conv(const Tensor& x,
                        const Tensor& weight,
                        Tensor& out,
                        aclrtStream stream) {
    if (x.dtype() != DType::Float16 || weight.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("linear_causal_conv requires fp16 tensors");
    }
    if (x.shape().size() != 2 || out.shape() != x.shape()) {
        throw std::runtime_error("linear_causal_conv x/out must be same [T, C] shape");
    }
    if (!((weight.shape().size() == 2 && weight.shape()[0] == x.shape()[1] && weight.shape()[1] == 4) ||
          (weight.shape().size() == 3 && weight.shape()[0] == x.shape()[1] && weight.shape()[1] == 1 && weight.shape()[2] == 4))) {
        throw std::runtime_error("linear_causal_conv weight must be [C,4] or [C,1,4]");
    }

    AclTensorHandle hx, hw, ho;
    make_acl_tensor(x, hx);
    make_acl_tensor(out, ho);

    if (weight.shape().size() == 2) {
        make_acl_tensor(weight, hw);
    } else {
        hw.storage_dims = weight.shape();
        hw.view_dims = {weight.shape()[0], weight.shape()[2]};
        hw.strides = {weight.shape()[2], 1};
        hw.tensor = aclCreateTensor(
            hw.view_dims.data(), hw.view_dims.size(), to_acl_dtype(weight.dtype()),
            hw.strides.data(), 0, ACL_FORMAT_ND,
            hw.storage_dims.data(), hw.storage_dims.size(),
            weight.data());
        if (hw.tensor == nullptr) {
            throw std::runtime_error("aclCreateTensor returned null for linear_causal_conv weight view");
        }
    }

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnLinearCausalConvCustomGetWorkspaceSize(hx.tensor, hw.tensor, ho.tensor,
                                                           &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnLinearCausalConvCustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnLinearCausalConvCustom", ws_size, executor, stream, aclnnLinearCausalConvCustom);
}

void linear_causal_conv_step(const Tensor& x,
                             const Tensor& weight_t,
                             Tensor& out,
                             aclrtStream stream) {
    if (x.dtype() != DType::Float16 || weight_t.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("linear_causal_conv_step requires fp16 tensors");
    }
    if (x.shape().size() != 2 || x.shape()[0] != 4) {
        throw std::runtime_error("linear_causal_conv_step x must be [4, C]");
    }
    const int64_t C = x.shape()[1];
    if (weight_t.shape() != std::vector<int64_t>{4, C}) {
        throw std::runtime_error("linear_causal_conv_step weight_t must be [4, C]");
    }
    if (out.shape() != std::vector<int64_t>{1, C}) {
        throw std::runtime_error("linear_causal_conv_step out must be [1, C]");
    }

    AclTensorHandle hx, hw, ho;
    make_acl_tensor(x, hx);
    make_acl_tensor(weight_t, hw);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnLinearCausalConvStepCustomGetWorkspaceSize(hx.tensor, hw.tensor, ho.tensor,
                                                                &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnLinearCausalConvStepCustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnLinearCausalConvStepCustom", ws_size, executor, stream, aclnnLinearCausalConvStepCustom);
}

void matmul_w4a16(const Tensor& x,
                  const Tensor& w_int8,
                  const Tensor& scales,
                  Tensor& out,
                  aclrtStream stream) {
    // GPTQ W4A16 matmul (M=1 fast path). The weight has been pre-unpacked
    // and block-packed by output-column chunks to match the custom kernel's 8-core split.
    if (x.dtype() != DType::Float16) {
        throw std::runtime_error("matmul_w4a16 x must be fp16");
    }
    if (w_int8.dtype() != DType::Int8) {
        throw std::runtime_error("matmul_w4a16 w must be int8 (pre-unpacked from GPTQ)");
    }
    if (scales.dtype() != DType::Float16) {
        throw std::runtime_error("matmul_w4a16 scales must be fp16");
    }
    if (out.dtype() != DType::Float16) {
        throw std::runtime_error("matmul_w4a16 out must be fp16");
    }
    if (x.shape().size() != 2 || x.shape()[0] != 1) {
        throw std::runtime_error("matmul_w4a16 x must be [1, K]");
    }
    const int64_t K = x.shape()[1];
    constexpr int64_t G = 128;
    if (scales.shape().size() != 2) {
        throw std::runtime_error("matmul_w4a16 scales must be 2D");
    }
    const int64_t N = scales.shape()[1];
    if (w_int8.shape().size() != 2) {
        throw std::runtime_error("matmul_w4a16 w must be [packed_rows, tile_len]");
    }
    const int64_t tileLen = w_int8.shape()[1];
    if (tileLen <= 0 || tileLen > 448 || tileLen % 16 != 0) {
        throw std::runtime_error("matmul_w4a16 tile_len must be a multiple of 16 and <= 448");
    }
    if (K % G != 0 || N % 128 != 0) {
        throw std::runtime_error("matmul_w4a16 K and N must be multiples of 128");
    }
    const int64_t blockLen = (N + 7) / 8;
    const int64_t tilesPerBlock = (blockLen + tileLen - 1) / tileLen;
    const int64_t packedRows = (K / G) * 8 * tilesPerBlock * G;
    if (w_int8.shape()[0] != packedRows) {
        throw std::runtime_error("matmul_w4a16 w packed_rows mismatch");
    }
    if (scales.shape() != std::vector<int64_t>{K / G, N}) {
        throw std::runtime_error("matmul_w4a16 scales must be [K/128, N]");
    }
    if (out.shape() != std::vector<int64_t>{1, N}) {
        throw std::runtime_error("matmul_w4a16 out must be [1, N]");
    }

    AclTensorHandle hx, hw, hs, ho;
    make_acl_tensor(x, hx);
    make_acl_tensor(w_int8, hw);
    make_acl_tensor(scales, hs);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMatmulW4a16CustomGetWorkspaceSize(hx.tensor, hw.tensor, hs.tensor,
                                                       ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnMatmulW4a16CustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnMatmulW4a16Custom", ws_size, executor, stream, aclnnMatmulW4a16Custom);
}

void matmul_w8a8_i32(const Tensor& x,
                     const Tensor& w_int8,
                     Tensor& out,
                     aclrtStream stream) {
    if (x.dtype() != DType::Int8) {
        throw std::runtime_error("matmul_w8a8_i32 x must be int8");
    }
    if (w_int8.dtype() != DType::Int8) {
        throw std::runtime_error("matmul_w8a8_i32 w must be int8");
    }
    if (out.dtype() != DType::Int32) {
        throw std::runtime_error("matmul_w8a8_i32 out must be int32");
    }
    if (x.shape().size() != 2 || x.shape()[0] != 1) {
        throw std::runtime_error("matmul_w8a8_i32 x must be [1, K]");
    }
    if (w_int8.shape().size() != 2 || w_int8.shape()[0] != x.shape()[1]) {
        throw std::runtime_error("matmul_w8a8_i32 w must be [K, N]");
    }
    if (out.shape() != std::vector<int64_t>{1, w_int8.shape()[1]}) {
        throw std::runtime_error("matmul_w8a8_i32 out must be [1, N]");
    }

    AclTensorHandle hx, hw, ho;
    make_acl_tensor(x, hx);
    make_acl_tensor(w_int8, hw);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMatmulW8a8I32CustomGetWorkspaceSize(hx.tensor, hw.tensor,
                                                         ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnMatmulW8a8I32CustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnMatmulW8a8I32Custom", ws_size, executor, stream, aclnnMatmulW8a8I32Custom);
}

void w8a8_quantize(const Tensor& x,
                   Tensor& x_int8,
                   Tensor& x_scale,
                   aclrtStream stream) {
    if (x.dtype() != DType::Float16) {
        throw std::runtime_error("w8a8_quantize x must be fp16");
    }
    if (x_int8.dtype() != DType::Int8) {
        throw std::runtime_error("w8a8_quantize x_int8 must be int8");
    }
    if (x_scale.dtype() != DType::Float16) {
        throw std::runtime_error("w8a8_quantize x_scale must be fp16");
    }
    if (x.shape().size() != 2 || x.shape()[0] != 1) {
        throw std::runtime_error("w8a8_quantize x must be [1, K]");
    }
    if (x_int8.shape() != x.shape()) {
        throw std::runtime_error("w8a8_quantize x_int8 must match x shape");
    }
    if (x_scale.shape() != std::vector<int64_t>{1}) {
        throw std::runtime_error("w8a8_quantize x_scale must be [1]");
    }

    AclTensorHandle hx, hxq, hs;
    make_acl_tensor(x, hx);
    make_acl_tensor(x_int8, hxq);
    make_acl_tensor(x_scale, hs);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnW8a8QuantizeCustomGetWorkspaceSize(hx.tensor, hxq.tensor, hs.tensor,
                                                        &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnW8a8QuantizeCustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnW8a8QuantizeCustom", ws_size, executor, stream, aclnnW8a8QuantizeCustom);
}

void w8a8_dequant(const Tensor& acc,
                  const Tensor& x_scale,
                  const Tensor& w_scale,
                  Tensor& out,
                  aclrtStream stream) {
    if (acc.dtype() != DType::Int32 || x_scale.dtype() != DType::Float16 ||
        w_scale.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("w8a8_dequant requires int32 acc, fp16 scales, fp16 out");
    }
    if (acc.shape().size() != 2 || acc.shape()[0] != 1) {
        throw std::runtime_error("w8a8_dequant acc must be [1, N]");
    }
    const int64_t N = acc.shape()[1];
    if (x_scale.shape() != std::vector<int64_t>{1}) {
        throw std::runtime_error("w8a8_dequant x_scale must be [1]");
    }
    if (w_scale.shape() != std::vector<int64_t>{N}) {
        throw std::runtime_error("w8a8_dequant w_scale must be [N]");
    }
    if (out.shape() != std::vector<int64_t>{1, N}) {
        throw std::runtime_error("w8a8_dequant out must be [1, N]");
    }

    AclTensorHandle ha, hxs, hws, ho;
    make_acl_tensor(acc, ha);
    make_acl_tensor(x_scale, hxs);
    make_acl_tensor(w_scale, hws);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnW8a8DequantCustomGetWorkspaceSize(ha.tensor, hxs.tensor, hws.tensor,
                                                       ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnW8a8DequantCustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnW8a8DequantCustom", ws_size, executor, stream, aclnnW8a8DequantCustom);
}

void linear_gated_delta_rule(const Tensor& mixed,
                             const Tensor& beta,
                             const Tensor& decay,
                             Tensor& scratch,
                             Tensor& out,
                             aclrtStream stream) {
    if (mixed.dtype() != DType::Float16 || beta.dtype() != DType::Float16 ||
        decay.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("linear_gated_delta_rule requires fp16 tensors");
    }
    if (scratch.dtype() != DType::Float32) {
        throw std::runtime_error("linear_gated_delta_rule scratch must be fp32");
    }
    constexpr int64_t kScratchElems = 8 * (128 * 128 + 5 * 128);
    if (scratch.shape().size() != 1 || scratch.shape()[0] != kScratchElems) {
        throw std::runtime_error("linear_gated_delta_rule scratch must be [136192] fp32");
    }
    if (mixed.shape().size() != 2 || mixed.shape()[1] != 6144) {
        throw std::runtime_error("linear_gated_delta_rule mixed shape must be [T, 6144]");
    }
    if (beta.shape().size() != 2 || beta.shape()[1] != 16 || beta.shape()[0] != mixed.shape()[0]) {
        throw std::runtime_error("linear_gated_delta_rule beta shape must be [T, 16]");
    }
    if (decay.shape() != beta.shape()) {
        throw std::runtime_error("linear_gated_delta_rule decay shape must match beta");
    }
    if (out.shape().size() != 2 || out.shape()[1] != 2048 || out.shape()[0] != mixed.shape()[0]) {
        throw std::runtime_error("linear_gated_delta_rule out shape must be [T, 2048]");
    }

    AclTensorHandle hm, hb, hd, hs, ho;
    make_acl_tensor(mixed, hm);
    make_acl_tensor(beta, hb);
    make_acl_tensor(decay, hd);
    make_acl_tensor(scratch, hs);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnLinearGatedDeltaRuleCustomGetWorkspaceSize(hm.tensor, hb.tensor, hd.tensor, hs.tensor,
                                                               ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnLinearGatedDeltaRuleCustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnLinearGatedDeltaRuleCustom", ws_size, executor, stream, aclnnLinearGatedDeltaRuleCustom);
}

void linear_gated_delta_rule_step(const Tensor& mixed,
                                  const Tensor& beta,
                                  const Tensor& decay,
                                  Tensor& state,
                                  Tensor& scratch,
                                  Tensor& out,
                                  aclrtStream stream) {
    if (mixed.dtype() != DType::Float16 || beta.dtype() != DType::Float16 ||
        decay.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("linear_gated_delta_rule_step requires fp16 mixed/beta/decay/out");
    }
    if (state.dtype() != DType::Float32 || scratch.dtype() != DType::Float32) {
        throw std::runtime_error("linear_gated_delta_rule_step state/scratch must be fp32");
    }
    if (mixed.shape() != std::vector<int64_t>{1, 6144}) {
        throw std::runtime_error("linear_gated_delta_rule_step mixed must be [1, 6144]");
    }
    if (beta.shape() != std::vector<int64_t>{1, 16}) {
        throw std::runtime_error("linear_gated_delta_rule_step beta must be [1, 16]");
    }
    if (decay.shape() != std::vector<int64_t>{1, 16}) {
        throw std::runtime_error("linear_gated_delta_rule_step decay must be [1, 16]");
    }
    if (state.shape() != std::vector<int64_t>{16, 128, 128}) {
        throw std::runtime_error("linear_gated_delta_rule_step state must be [16, 128, 128] fp32");
    }
    constexpr int64_t kScratchElems = 8 * 6 * 128;
    if (scratch.shape().size() != 1 || scratch.shape()[0] != kScratchElems) {
        throw std::runtime_error("linear_gated_delta_rule_step scratch must be [6144] fp32");
    }
    if (out.shape() != std::vector<int64_t>{1, 2048}) {
        throw std::runtime_error("linear_gated_delta_rule_step out must be [1, 2048]");
    }

    AclTensorHandle hm, hb, hd, hstate, hscratch, ho;
    make_acl_tensor(mixed, hm);
    make_acl_tensor(beta, hb);
    make_acl_tensor(decay, hd);
    make_acl_tensor(state, hstate);
    make_acl_tensor(scratch, hscratch);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnLinearGatedDeltaRuleStepCustomGetWorkspaceSize(hm.tensor, hb.tensor, hd.tensor,
                                                                    hstate.tensor, hscratch.tensor,
                                                                    ho.tensor,
                                                                    &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnLinearGatedDeltaRuleStepCustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnLinearGatedDeltaRuleStepCustom", ws_size, executor, stream, aclnnLinearGatedDeltaRuleStepCustom);
}

void gated_rms_norm_z(const Tensor& core,
                      const Tensor& z_silu,
                      const Tensor& gamma,
                      Tensor& out,
                      aclrtStream stream) {
    if (core.dtype() != DType::Float16 || z_silu.dtype() != DType::Float16 ||
        gamma.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("gated_rms_norm_z requires fp16 tensors");
    }
    if (core.shape().size() != 2 || core.shape()[1] != 2048) {
        throw std::runtime_error("gated_rms_norm_z core shape must be [T, 2048]");
    }
    if (z_silu.shape() != core.shape()) {
        throw std::runtime_error("gated_rms_norm_z z_silu shape must match core");
    }
    if (out.shape() != core.shape()) {
        throw std::runtime_error("gated_rms_norm_z out shape must match core");
    }
    if (gamma.shape().size() != 1 || gamma.shape()[0] != 128) {
        throw std::runtime_error("gated_rms_norm_z gamma shape must be [128]");
    }

    AclTensorHandle hc, hz, hg, ho;
    make_acl_tensor(core, hc);
    make_acl_tensor(z_silu, hz);
    make_acl_tensor(gamma, hg);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnGatedRmsNormZCustomGetWorkspaceSize(hc.tensor, hz.tensor, hg.tensor,
                                                        ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnGatedRmsNormZCustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnGatedRmsNormZCustom", ws_size, executor, stream, aclnnGatedRmsNormZCustom);
}

// ---- Vision encoder building blocks (Conv2d, GELU, BatchMatMul, linear+bias) ----

void conv2d(const Tensor& input,
            const Tensor& weight,
            const Tensor* bias,
            const std::vector<int64_t>& stride,
            const std::vector<int64_t>& padding,
            Tensor& out,
            aclrtStream stream) {
    if (input.dtype() != DType::Float16 || weight.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("conv2d requires fp16 tensors");
    }
    if (bias != nullptr && bias->dtype() != DType::Float16) {
        throw std::runtime_error("conv2d bias must be fp16");
    }
    if (input.shape().size() != 4 || weight.shape().size() != 4 || out.shape().size() != 4) {
        throw std::runtime_error("conv2d expects 4D NCHW tensors");
    }
    if (stride.size() != 2 || padding.size() != 2) {
        throw std::runtime_error("conv2d stride/padding must be size 2 (H, W)");
    }

    auto make_nchw = [](const Tensor& t, AclTensorHandle& h) {
        h.view_dims = t.shape();
        h.storage_dims = t.shape();
        h.strides.assign(t.shape().size(), 1);
        for (int i = static_cast<int>(t.shape().size()) - 2; i >= 0; --i) {
            h.strides[i] = h.strides[i + 1] * t.shape()[i + 1];
        }
        h.tensor = aclCreateTensor(
            h.view_dims.data(), h.view_dims.size(), to_acl_dtype(t.dtype()),
            h.strides.data(), 0, ACL_FORMAT_NCHW,
            h.storage_dims.data(), h.storage_dims.size(),
            t.data());
        if (h.tensor == nullptr) throw std::runtime_error("conv2d aclCreateTensor (NCHW) returned null");
    };

    AclTensorHandle hi, hw, hb, ho;
    make_nchw(input, hi);
    make_nchw(weight, hw);
    make_nchw(out, ho);
    if (bias != nullptr) {
        make_acl_tensor(*bias, hb);  // bias is 1D ND
    }

    aclIntArray* stride_arr = aclCreateIntArray(stride.data(), stride.size());
    aclIntArray* pad_arr = aclCreateIntArray(padding.data(), padding.size());
    std::vector<int64_t> dilation_vec{1, 1};
    aclIntArray* dilation_arr = aclCreateIntArray(dilation_vec.data(), dilation_vec.size());
    std::vector<int64_t> output_pad_vec{0, 0};
    aclIntArray* output_pad_arr = aclCreateIntArray(output_pad_vec.data(), output_pad_vec.size());

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    constexpr int8_t kCubeMathType = 1;  // ALLOW_FP32_DOWN_PRECISION
    auto ret = aclnnConvolutionGetWorkspaceSize(
        hi.tensor, hw.tensor,
        bias != nullptr ? hb.tensor : nullptr,
        stride_arr, pad_arr, dilation_arr,
        /*transposed=*/false, output_pad_arr, /*groups=*/1,
        ho.tensor, kCubeMathType, &ws_size, &executor);
    if (ret != 0) {
        aclDestroyIntArray(stride_arr); aclDestroyIntArray(pad_arr);
        aclDestroyIntArray(dilation_arr); aclDestroyIntArray(output_pad_arr);
        throw std::runtime_error("aclnnConvolutionGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnConvolution", ws_size, executor, stream, aclnnConvolution);
    aclDestroyIntArray(stride_arr);
    aclDestroyIntArray(pad_arr);
    aclDestroyIntArray(dilation_arr);
    aclDestroyIntArray(output_pad_arr);
}

void gelu(const Tensor& self, bool tanh_approx, Tensor& out, aclrtStream stream) {
    if (self.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("gelu requires fp16 tensors");
    }
    if (self.shape() != out.shape()) {
        throw std::runtime_error("gelu shape mismatch");
    }
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    const int64_t approximate = tanh_approx ? 1 : 0;
    auto ret = aclnnGeluV2GetWorkspaceSize(hs.tensor, approximate, ho.tensor, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnGeluV2GetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnGeluV2", ws_size, executor, stream, aclnnGeluV2);
}

void batch_matmul(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream) {
    if (a.dtype() != DType::Float16 || b.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("batch_matmul requires fp16 tensors");
    }
    if (a.shape().size() < 3 || b.shape().size() != a.shape().size() || out.shape().size() != a.shape().size()) {
        throw std::runtime_error("batch_matmul tensors must share rank >= 3");
    }
    const int64_t r = a.shape().size();
    if (a.shape()[r - 1] != b.shape()[r - 2]) {
        throw std::runtime_error("batch_matmul inner dims mismatch");
    }
    if (out.shape()[r - 2] != a.shape()[r - 2] || out.shape()[r - 1] != b.shape()[r - 1]) {
        throw std::runtime_error("batch_matmul out shape mismatch");
    }
    for (int64_t i = 0; i < r - 2; ++i) {
        if (a.shape()[i] != b.shape()[i] || a.shape()[i] != out.shape()[i]) {
            throw std::runtime_error("batch_matmul batch dims must broadcast-match");
        }
    }

    AclTensorHandle ha, hb, ho;
    make_acl_tensor(a, ha);
    make_acl_tensor(b, hb);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    constexpr int8_t kCubeMathType = 1;
    auto ret = aclnnBatchMatMulGetWorkspaceSize(ha.tensor, hb.tensor, ho.tensor, kCubeMathType, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnBatchMatMulGetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnBatchMatMul", ws_size, executor, stream, aclnnBatchMatMul);
}

void linear_bias(const Tensor& x,
                 const Tensor& w,
                 const Tensor* bias,
                 Tensor& out,
                 aclrtStream stream) {
    matmul_b_transposed(x, w, out, stream);
    if (bias == nullptr) return;
    // Broadcast add: bias is [N], out is [..., N]. Use aclnnInplaceAdd which
    // is the dedicated in-place add API (avoids the kernel-cache corruption
    // we saw when aliasing input and output through plain aclnnAdd).
    if (bias->dtype() != DType::Float16) throw std::runtime_error("linear_bias bias must be fp16");
    if (bias->shape().size() != 1 || bias->shape()[0] != out.shape().back()) {
        throw std::runtime_error("linear_bias bias must be [N] matching out last dim");
    }
    AclTensorHandle ho, hb;
    make_acl_tensor(out, ho);
    make_acl_tensor(*bias, hb);
    float alpha_value = 1.0f;
    aclScalar* alpha = aclCreateScalar(&alpha_value, ACL_FLOAT);
    if (alpha == nullptr) throw std::runtime_error("linear_bias aclCreateScalar failed");
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceAddGetWorkspaceSize(ho.tensor, hb.tensor, alpha, &ws_size, &executor);
    if (ret != 0) {
        aclDestroyScalar(alpha);
        throw std::runtime_error("linear_bias aclnnInplaceAddGetWorkspaceSize failed: " + std::to_string(ret));
    }
    try {
        run_op("aclnnInplaceAdd_bias", ws_size, executor, stream, aclnnInplaceAdd);
    } catch (...) {
        aclDestroyScalar(alpha);
        throw;
    }
    aclDestroyScalar(alpha);
}

void permute(const Tensor& self,
             const std::vector<int64_t>& dims,
             Tensor& out,
             aclrtStream stream) {
    if (self.dtype() != out.dtype()) throw std::runtime_error("permute dtype mismatch");
    if (dims.size() != self.shape().size()) throw std::runtime_error("permute dims rank mismatch");
    for (size_t i = 0; i < dims.size(); ++i) {
        if (out.shape()[i] != self.shape()[dims[i]]) {
            throw std::runtime_error("permute out shape mismatch");
        }
    }
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    aclIntArray* dims_arr = aclCreateIntArray(dims.data(), dims.size());
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPermuteGetWorkspaceSize(hs.tensor, dims_arr, ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        aclDestroyIntArray(dims_arr);
        throw std::runtime_error("aclnnPermuteGetWorkspaceSize failed: " + std::to_string(ret));
    }
    try {
        run_op("aclnnPermute", ws_size, executor, stream, aclnnPermute);
    } catch (...) {
        aclDestroyIntArray(dims_arr);
        throw;
    }
    aclDestroyIntArray(dims_arr);
}

void muls(const Tensor& self, float scalar, Tensor& out, aclrtStream stream) {
    if (self.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("muls requires fp16 tensors");
    }
    if (self.shape() != out.shape()) throw std::runtime_error("muls shape mismatch");
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    aclScalar* s = aclCreateScalar(&scalar, ACL_FLOAT);
    if (s == nullptr) throw std::runtime_error("muls aclCreateScalar failed");
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulsGetWorkspaceSize(hs.tensor, s, ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        aclDestroyScalar(s);
        throw std::runtime_error("aclnnMulsGetWorkspaceSize failed: " + std::to_string(ret));
    }
    try {
        run_op("aclnnMuls", ws_size, executor, stream, aclnnMuls);
    } catch (...) {
        aclDestroyScalar(s);
        throw;
    }
    aclDestroyScalar(s);
}

void mean(const Tensor& self,
          const std::vector<int64_t>& dims,
          bool keep_dim,
          Tensor& out,
          aclrtStream stream) {
    if (self.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("mean requires fp16 tensors");
    }
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    aclIntArray* dim_arr = aclCreateIntArray(dims.data(), dims.size());
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMeanGetWorkspaceSize(hs.tensor, dim_arr, keep_dim,
                                         to_acl_dtype(self.dtype()),
                                         ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        aclDestroyIntArray(dim_arr);
        throw std::runtime_error("aclnnMeanGetWorkspaceSize failed: " + std::to_string(ret));
    }
    try {
        run_op("aclnnMean", ws_size, executor, stream, aclnnMean);
    } catch (...) {
        aclDestroyIntArray(dim_arr);
        throw;
    }
    aclDestroyIntArray(dim_arr);
}

}  // namespace minicpmo
