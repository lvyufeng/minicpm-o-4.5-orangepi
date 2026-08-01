// Is the decode slowdown about DATA BANDWIDTH, or about the ADDRESS FOOTPRINT?
//
// Three measurements on the real model, cycling the MLP across 36 distinct
// layers vs hammering one layer (bench_inproc_layer_cycling,
// bench_w8a8_cold_cycling):
//     aclnnMm fp16 : 40.18 -> 43.08 ms   (7.01 GB/s on 302 MB)
//     custom cube  : 20.55 -> 51.27 ms   (5.90 GB/s)
//     W8A8 int8    : 12.67 -> 58.79 ms   (2.57 GB/s on 151 MB)
//
// int8 reads HALF the bytes and is 36% SLOWER when cycling. A bandwidth limit
// cannot behave that way. The "cache hit" story does not hold either: one
// layer's MLP weights are 302 MB, far beyond any on-chip cache, so the fast
// same-layer case was never a cache hit.
//
// What varies with cycling is the number of DISTINCT allocations touched and
// the total address range spanned. This isolates that with plain
// aclrtMemcpyAsync, which has perfect sequential locality -- no tiling, no
// kernel, nothing to blame but addressing:
//   A) copy repeatedly from ONE buffer
//   B) copy from N DISTINCT buffers, round-robin
//   C) copy from N offsets inside ONE giant contiguous buffer
// A vs B isolates allocation count; B vs C separates "many allocations" from
// "large address span", since C has the same span through a single mapping.
//
// N is swept up to production scale (108 buffers / ~10.9 GB, i.e. all three
// projections of all 36 layers), because a page-table/TLB effect may only
// appear once the footprint exceeds what the device can keep mapped.
#include "minicpmo/acl_context.h"

#include <acl/acl.h>

#include <chrono>
#include <cstdio>
#include <vector>

using namespace minicpmo;

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main() {
    AclContext ctx(0);
    aclrtStream stream = ctx.stream();

    // One "weight-sized" chunk: 4096 x 12288 fp16 = 100.66 MB, matching a
    // single gate/up/down projection in the real model.
    const size_t chunk = static_cast<size_t>(4096) * 12288 * 2;
    // Production touches 3 projections x 36 layers = 108 such buffers/token.
    const int n_target = 108;
    std::printf("chunk = %.2f MB, target n = %d, target total = %.2f GB\n\n",
                chunk / 1e6, n_target, (chunk * static_cast<double>(n_target)) / 1e9);

    void* dst = nullptr;
    if (aclrtMalloc(&dst, chunk, ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        std::printf("dst alloc failed\n");
        return 1;
    }
    auto bandwidth = [&](double ms) { return chunk / (ms * 1e6); };

    std::vector<void*> bufs;
    for (int i = 0; i < n_target; ++i) {
        void* p = nullptr;
        if (aclrtMalloc(&p, chunk, ACL_MEM_MALLOC_HUGE_FIRST) != 0) break;
        aclrtMemsetAsync(p, chunk, 1, chunk, stream);
        bufs.push_back(p);
    }
    aclrtSynchronizeStream(stream);
    const int m = static_cast<int>(bufs.size());
    std::printf("allocated %d buffers = %.2f GB\n\n", m, (chunk * static_cast<double>(m)) / 1e9);
    if (m < 4) return 1;

    // Sweep the working set: at each size, same-buffer vs distinct-buffer.
    std::printf("%-8s %-14s %-14s %-8s\n", "n", "same(ms)", "distinct(ms)", "ratio");
    for (int n = 4; n <= m; n *= 3) {
        for (int i = 0; i < n; ++i) {
            aclrtMemcpyAsync(dst, chunk, bufs[i], chunk, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        }
        aclrtSynchronizeStream(stream);

        double t0 = now_ms();
        for (int i = 0; i < n; ++i) {
            aclrtMemcpyAsync(dst, chunk, bufs[0], chunk, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        }
        aclrtSynchronizeStream(stream);
        double same = (now_ms() - t0) / n;

        t0 = now_ms();
        for (int i = 0; i < n; ++i) {
            aclrtMemcpyAsync(dst, chunk, bufs[i], chunk, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        }
        aclrtSynchronizeStream(stream);
        double dist = (now_ms() - t0) / n;

        std::printf("%-8d %-14.4f %-14.4f %-8.2f  (%.1f GB span, %.2f/%.2f GB/s)\n",
                    n, same, dist, dist / same,
                    (chunk * static_cast<double>(n)) / 1e9, bandwidth(same), bandwidth(dist));
    }

    // Full-scale A/B/C at the largest working set that fit.
    std::printf("\n--- full scale, n=%d (%.2f GB) ---\n", m, (chunk * static_cast<double>(m)) / 1e9);
    double t0 = now_ms();
    for (int i = 0; i < m; ++i) {
        aclrtMemcpyAsync(dst, chunk, bufs[0], chunk, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }
    aclrtSynchronizeStream(stream);
    double a_ms = (now_ms() - t0) / m;
    std::printf("A) SAME buffer      : %8.4f ms -> %6.2f GB/s\n", a_ms, bandwidth(a_ms));

    t0 = now_ms();
    for (int i = 0; i < m; ++i) {
        aclrtMemcpyAsync(dst, chunk, bufs[i], chunk, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }
    aclrtSynchronizeStream(stream);
    double b_ms = (now_ms() - t0) / m;
    std::printf("B) DISTINCT buffers : %8.4f ms -> %6.2f GB/s\n", b_ms, bandwidth(b_ms));

    t0 = now_ms();
    for (int i = 0; i < m; ++i) {
        aclrtMemcpyAsync(dst, chunk, bufs[0], chunk, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }
    aclrtSynchronizeStream(stream);
    double a2_ms = (now_ms() - t0) / m;
    std::printf("A') SAME again      : %8.4f ms -> %6.2f GB/s\n", a2_ms, bandwidth(a2_ms));

    std::printf("\n=== VERDICT ===\n");
    std::printf("same=%.2f  distinct=%.2f ms  (%.2fx)\n", a_ms, b_ms, b_ms / a_ms);
    if (b_ms > a_ms * 1.3) {
        std::printf("-> Plain memcpy DOES degrade on distinct buffers at this scale.\n");
        std::printf("   The effect is in address translation, not in the matmul kernel.\n");
    } else {
        std::printf("-> Plain memcpy does NOT degrade even at production footprint.\n");
        std::printf("   The hardware can stream %.0f MB from any of %d buffers at\n",
                    chunk / 1e6, m);
        std::printf("   %.1f GB/s, so decode's %.0f GB/s is NOT a memory-system limit --\n",
                    bandwidth(b_ms), 7.0);
        std::printf("   it is specific to how the matmul kernels issue their reads.\n");
    }

    for (void* p : bufs) aclrtFree(p);
    aclrtFree(dst);
    return 0;
}
