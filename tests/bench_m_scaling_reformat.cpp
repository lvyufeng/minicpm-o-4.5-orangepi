// Tests the weight-reformat theory by its strongest prediction: M-invariance.
//
// The clue: aclnnMm requests a 101 MB workspace for a 1x4096 @ 4096x12288
// product -- exactly the weight's size. That looks like a per-call reformat of
// the ND weight into the Cube unit's FRACTAL_NZ tiling. Traffic would then be
// read 101 + write 101 + read 101 = ~300 MB, i.e. ~13.6 ms at the 22.1 GB/s
// the hardware sustains (bench_alloc_footprint_memcpy), matching the 11.7 ms
// measured against a 4.55 ms weight-streaming floor.
//
// aclnnNpuFormatCast, which would let weights be pre-converted once at load
// time, returns 161002 (unsupported) on this 310B, so the theory cannot be
// tested that way. But it makes a sharp, cheap prediction instead:
//
//   If time is dominated by reformatting the WEIGHT, it is independent of M.
//   M=1 and M=64 reformat the same 101 MB; only the (tiny) Cube work grows.
//   So cost should be FLAT in M, then rise once real compute takes over.
//
//   If instead time were compute- or activation-bound, it would scale with M
//   from the start.
//
// Supporting evidence already in hand: aclnnMm barely cares whether weights
// are cold (40.18 same-layer vs 43.08 ms cycling, 1.07x), which is what a
// fixed per-call reformat cost looks like -- it pays the same price either way.
//
// Also reports workspace size per shape: if workspace tracks K*N*2 exactly
// across shapes, that pins it to the weight rather than to the output.
#include "minicpmo/acl_context.h"

#include <acl/acl.h>
#include <aclnnop/aclnn_mm.h>

#include <chrono>
#include <cstdio>
#include <vector>

using namespace minicpmo;

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

constexpr int8_t kCubeMathType = 1;

aclTensor* make_nd(void* data, const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size());
    int64_t acc = 1;
    for (size_t i = dims.size(); i-- > 0;) {
        strides[i] = acc;
        acc *= dims[i];
    }
    return aclCreateTensor(dims.data(), dims.size(), ACL_FLOAT16, strides.data(), 0,
                           ACL_FORMAT_ND, dims.data(), dims.size(), data);
}

}  // namespace

int main() {
    AclContext ctx(0);
    aclrtStream stream = ctx.stream();

    const int64_t K = 4096, N = 12288;
    const size_t wbytes = static_cast<size_t>(K) * static_cast<size_t>(N) * 2;
    const int n_layers = 24;   // enough distinct buffers to keep reads cold

    std::vector<void*> w;
    for (int i = 0; i < n_layers; ++i) {
        void* p = nullptr;
        if (aclrtMalloc(&p, wbytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0) break;
        aclrtMemsetAsync(p, wbytes, 0x34, wbytes, stream);
        w.push_back(p);
    }
    aclrtSynchronizeStream(stream);
    const int m = static_cast<int>(w.size());
    std::printf("%d weights of %.2f MB, K=%ld N=%ld\n", m, wbytes / 1e6,
                static_cast<long>(K), static_cast<long>(N));
    std::printf("weight bytes at 22.1 GB/s (memcpy-measured) = %.2f ms\n\n",
                wbytes / 22.1e6);
    if (m < 2) return 1;

    const int64_t max_M = 64;
    void *x_dev = nullptr, *out_dev = nullptr;
    aclrtMalloc(&x_dev, static_cast<size_t>(max_M * K) * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&out_dev, static_cast<size_t>(max_M * N) * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    {
        std::vector<uint16_t> xh(static_cast<size_t>(max_M * K), 0x3c00u);
        aclrtMemcpy(x_dev, xh.size() * 2, xh.data(), xh.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    std::printf("%-6s %-12s %-14s %-12s %-10s\n",
                "M", "ms/call", "workspace(MB)", "GFLOP/s", "vs M=1");
    double base = 0.0;
    for (int64_t M : {int64_t{1}, int64_t{2}, int64_t{4}, int64_t{8},
                      int64_t{16}, int64_t{32}, int64_t{64}}) {
        aclTensor* ta = make_nd(x_dev, {M, K});
        aclTensor* tc = make_nd(out_dev, {M, N});

        uint64_t ws_seen = 0;
        auto once = [&](int i) {
            aclTensor* tb = make_nd(w[static_cast<size_t>(i % m)], {K, N});
            uint64_t ws = 0;
            aclOpExecutor* ex = nullptr;
            if (aclnnMmGetWorkspaceSize(ta, tb, tc, kCubeMathType, &ws, &ex) == 0) {
                ws_seen = ws;
                void* wsp = nullptr;
                if (ws) aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
                aclnnMm(wsp, ws, ex, stream);
                aclrtSynchronizeStream(stream);
                if (wsp) aclrtFree(wsp);
            }
            aclDestroyTensor(tb);
        };

        for (int i = 0; i < m; ++i) once(i);
        aclrtSynchronizeStream(stream);
        double t0 = now_ms();
        for (int i = 0; i < m; ++i) once(i);
        aclrtSynchronizeStream(stream);
        double ms = (now_ms() - t0) / m;

        if (M == 1) base = ms;
        const double gflops = (2.0 * static_cast<double>(M) * K * N) / (ms * 1e6);
        std::printf("%-6ld %-12.4f %-14.2f %-12.2f %-10.2fx\n",
                    static_cast<long>(M), ms, ws_seen / 1e6, gflops, ms / base);

        aclDestroyTensor(ta);
        aclDestroyTensor(tc);
    }

    std::printf("\n=== reading this ===\n");
    std::printf("Flat ms across M  -> cost is per-call WEIGHT handling (reformat).\n");
    std::printf("                     Decode wastes nearly all of it: M=1 pays the\n");
    std::printf("                     same price as M=64 but does 1/64 the work.\n");
    std::printf("Rising ms with M  -> cost is compute/activations, reformat theory dead.\n");

    for (void* p : w) aclrtFree(p);
    aclrtFree(x_dev);
    aclrtFree(out_dev);
    return 0;
}
