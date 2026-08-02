// Minimal repro/correctness check for aclnnMatmulCubeCustom. Verifies the
// custom cube-path matmul (M=16, K=128, N=16) against a reference host
// computation, and reports the returned status codes so operators can be
// spot-checked when ASCEND_CUSTOM_OPP_PATH configuration changes.
#include "minicpmo/acl_context.h"
#include "minicpmo/tensor.h"
#include "aclnn_matmul_cube_custom.h"

#include <acl/acl.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace minicpmo;

namespace {

uint16_t f32_to_f16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00);
    return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
}

float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    if (exp == 0 && mant == 0) { float f; uint32_t b = sign; std::memcpy(&f, &b, 4); return f; }
    uint32_t f32exp = exp - 15 + 127;
    uint32_t bits = sign | (f32exp << 23) | (mant << 13);
    float f; std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

int main() {
    AclContext ctx(0);
    aclrtStream stream = ctx.stream();

    const int64_t M = 16, K = 128, N = 16;  // M=16 (block-aligned), N=16 forces blockDim=1

    Tensor a({M, K}, DType::Float16); a.allocate();
    Tensor b({K, N}, DType::Float16); b.allocate();
    Tensor out({M, N}, DType::Float16); out.allocate();

    std::vector<uint16_t> a_host(static_cast<size_t>(M * K));
    std::vector<uint16_t> b_host(static_cast<size_t>(K * N));
    // Real fractional test values, both positive and negative, covering K=128
    // fully dense in both operands. i%7 and i%5 are cast to int64_t before the
    // subtraction -- doing the subtraction directly on the unsigned `size_t`
    // result of `%` underflows for i%7 in {0,1,2} (and i%5 in {0,1}), producing
    // a huge unsigned value that converts to +inf in fp16 instead of a small
    // negative fraction. That was a bug in this test harness, not the kernel:
    // it made earlier debugging sessions believe the cube kernel corrupted
    // dense rows containing negative values, when actually the "negative"
    // test inputs were never negative -- they were +infinity, and the
    // hardware's saturating cast to fp16-max on `inf * finite` was correct.
    for (size_t i = 0; i < a_host.size(); ++i) {
        a_host[i] = f32_to_f16(0.01f * static_cast<float>(static_cast<int64_t>(i % 7) - 3));
    }
    for (size_t i = 0; i < b_host.size(); ++i) {
        b_host[i] = f32_to_f16(0.01f * static_cast<float>(static_cast<int64_t>(i % 5) - 2));
    }
    a.copy_from_host(a_host.data(), a_host.size() * sizeof(uint16_t));
    b.copy_from_host(b_host.data(), b_host.size() * sizeof(uint16_t));

    auto make_tensor = [](const Tensor& t) -> aclTensor* {
        std::vector<int64_t> dims = t.shape();
        std::vector<int64_t> strides(dims.size(), 1);
        for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        return aclCreateTensor(dims.data(), dims.size(), ACL_FLOAT16,
                                strides.data(), 0, ACL_FORMAT_ND,
                                dims.data(), dims.size(), t.data());
    };

    aclTensor* ta = make_tensor(a);
    aclTensor* tb = make_tensor(b);
    aclTensor* tout = make_tensor(out);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMatmulCubeCustomGetWorkspaceSize(ta, tb, tout, &ws_size, &executor);
    std::printf("aclnnMatmulCubeCustomGetWorkspaceSize returned %d, ws_size=%lu\n",
                static_cast<int>(ret), static_cast<unsigned long>(ws_size));
    if (ret != 0) return 1;

    void* workspace = nullptr;
    if (ws_size > 0) aclrtMalloc(&workspace, ws_size, ACL_MEM_MALLOC_HUGE_FIRST);
    auto ret2 = aclnnMatmulCubeCustom(workspace, ws_size, executor, stream);
    aclrtSynchronizeStream(stream);
    if (workspace) aclrtFree(workspace);
    std::printf("aclnnMatmulCubeCustom returned %d\n", static_cast<int>(ret2));
    if (ret2 != 0) return 1;

    std::vector<uint16_t> out_host(static_cast<size_t>(M * N));
    out.copy_to_host(out_host.data(), out_host.size() * sizeof(uint16_t));

    // Reference: out[m, n] = sum_k a[m, k] * b[k, n]
    double max_abs_err = 0.0, max_rel_err = 0.0;
    int64_t worst_m = -1, worst_n = -1;
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                acc += static_cast<double>(f16_to_f32(a_host[static_cast<size_t>(m * K + k)])) *
                       static_cast<double>(f16_to_f32(b_host[static_cast<size_t>(k * N + n)]));
            }
            double got = f16_to_f32(out_host[static_cast<size_t>(m * N + n)]);
            double err = std::fabs(got - acc);
            if (err > max_abs_err) { max_abs_err = err; worst_m = m; worst_n = n; }
            if (std::fabs(acc) > 1e-6) max_rel_err = std::max(max_rel_err, err / std::fabs(acc));
        }
    }
    std::printf("max_abs_err=%.6f max_rel_err=%.6f at m=%lld n=%lld\n", max_abs_err, max_rel_err,
                static_cast<long long>(worst_m), static_cast<long long>(worst_n));
    std::printf(max_rel_err < 0.05 ? "PASS\n" : "FAIL\n");

    return 0;
}
