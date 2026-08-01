// Can we hand aclnnMm a weight already in FRACTAL_NZ and skip the reformat?
//
// The mechanism is pinned down (bench_m_scaling_reformat): aclnnMm asks for a
// workspace exactly the size of B when M < 16 (101.19 MB for 4096x12288 fp16)
// and 0 at M >= 16, and cost is FLAT from M=1 to M=8 (12.3 ms) before dropping
// to 7.8 ms at M=16. So below M=16 it re-formats the whole ND weight into the
// Cube's fractal tiling on every single call. Traffic is read 101 + write 101 +
// read 101 = ~300 MB, i.e. ~13.6 ms at the 22.1 GB/s the hardware sustains
// (bench_alloc_footprint_memcpy) -- matching the 12.3 ms measured, against a
// 4.55 ms floor for streaming the weight bytes once.
//
// aclnnNpuFormatCast, the supported way to pre-convert, returns 161002 on this
// 310B. But FRACTAL_NZ is a fixed, known layout, so we can build it on the host
// at load time instead. For a [rows, cols] fp16 matrix with C0 = 16:
//     storage = [cols/16, rows/16, 16, 16]
//     element (r, c) -> [c/16][r/16][r%16][c%16]
//
// If aclnnMm accepts an NZ B, the workspace should drop to 0 and the time
// toward 4.55 ms. Correctness is checked against the ND result on the same
// data, because a wrong layout guess would still "work" and just compute
// garbage. Timing uses distinct allocations so reads stay cold, per the
// weight-reuse lesson.
#include "minicpmo/acl_context.h"

#include <acl/acl.h>
#include <aclnnop/aclnn_mm.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace minicpmo;

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

constexpr int8_t kCubeMathType = 1;

uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) out = sign;
    else if (exp == 31) out = sign | 0x7f800000u | (mant << 13);
    else out = sign | ((exp + 112u) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

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

// Logical [K, N] view over storage laid out as FRACTAL_NZ.
aclTensor* make_nz(void* data, int64_t K, int64_t N, int format) {
    const std::vector<int64_t> view{K, N};
    const std::vector<int64_t> strides{N, 1};
    const std::vector<int64_t> storage{N / 16, K / 16, 16, 16};
    return aclCreateTensor(view.data(), view.size(), ACL_FLOAT16, strides.data(), 0,
                           static_cast<aclFormat>(format),
                           storage.data(), storage.size(), data);
}

}  // namespace

int main() {
    AclContext ctx(0);
    aclrtStream stream = ctx.stream();

    const int64_t K = 4096, N = 12288;
    const size_t wn = static_cast<size_t>(K) * static_cast<size_t>(N);
    const size_t wbytes = wn * 2;
    const int n_bufs = 24;

    std::printf("K=%ld N=%ld weight=%.2f MB\n", static_cast<long>(K),
                static_cast<long>(N), wbytes / 1e6);
    std::printf("streaming floor (22.1 GB/s measured) = %.2f ms\n\n", wbytes / 22.1e6);

    // Host ND weight, plus its NZ permutation.
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> nd_host(wn), nz_host(wn);
    std::vector<uint16_t> x_host(static_cast<size_t>(K));
    for (size_t i = 0; i < wn; ++i) nd_host[i] = f32_to_f16(dist(rng) * 0.05f);
    for (int64_t k = 0; k < K; ++k) x_host[static_cast<size_t>(k)] = f32_to_f16(dist(rng));

    const int64_t k1 = K / 16, n1 = N / 16;
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t n = 0; n < N; ++n) {
            const size_t src = static_cast<size_t>(k) * static_cast<size_t>(N) +
                               static_cast<size_t>(n);
            const size_t dst = ((static_cast<size_t>(n / 16) * static_cast<size_t>(k1)) +
                                static_cast<size_t>(k / 16)) * 256u +
                               static_cast<size_t>(k % 16) * 16u +
                               static_cast<size_t>(n % 16);
            nz_host[dst] = nd_host[src];
        }
    }
    std::printf("built NZ permutation: storage [%ld, %ld, 16, 16]\n\n",
                static_cast<long>(n1), static_cast<long>(k1));

    void *x_dev = nullptr, *o_nd = nullptr, *o_nz = nullptr;
    aclrtMalloc(&x_dev, static_cast<size_t>(K) * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&o_nd, static_cast<size_t>(N) * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&o_nz, static_cast<size_t>(N) * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(x_dev, x_host.size() * 2, x_host.data(), x_host.size() * 2,
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclTensor* ta = make_nd(x_dev, {1, K});
    aclTensor* tc_nd = make_nd(o_nd, {1, N});
    aclTensor* tc_nz = make_nd(o_nz, {1, N});

    // Reference: ND weight through the normal path.
    void* nd_dev = nullptr;
    aclrtMalloc(&nd_dev, wbytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(nd_dev, wbytes, nd_host.data(), wbytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclTensor* tb_nd = make_nd(nd_dev, {K, N});
    {
        uint64_t ws = 0;
        aclOpExecutor* ex = nullptr;
        auto r = aclnnMmGetWorkspaceSize(ta, tb_nd, tc_nd, kCubeMathType, &ws, &ex);
        std::printf("ND  B: ret=%d workspace=%.2f MB\n", r, ws / 1e6);
        void* wsp = nullptr;
        if (ws) aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnMm(wsp, ws, ex, stream);
        aclrtSynchronizeStream(stream);
        if (wsp) aclrtFree(wsp);
    }

    // Candidate NZ format enums: FRACTAL_NZ(29), FRACTAL_NZ_C0_16(50).
    void* nz_dev = nullptr;
    aclrtMalloc(&nz_dev, wbytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(nz_dev, wbytes, nz_host.data(), wbytes, ACL_MEMCPY_HOST_TO_DEVICE);

    int good_format = -1;
    for (int fmt : {29, 50}) {
        aclTensor* tb = make_nz(nz_dev, K, N, fmt);
        if (tb == nullptr) {
            std::printf("NZ  B (format %d): aclCreateTensor returned null\n", fmt);
            continue;
        }
        uint64_t ws = 0;
        aclOpExecutor* ex = nullptr;
        auto r = aclnnMmGetWorkspaceSize(ta, tb, tc_nz, kCubeMathType, &ws, &ex);
        if (r != 0) {
            std::printf("NZ  B (format %d): GetWorkspaceSize ret=%d (rejected)\n", fmt, r);
            aclDestroyTensor(tb);
            continue;
        }
        void* wsp = nullptr;
        if (ws) aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        auto r2 = aclnnMm(wsp, ws, ex, stream);
        aclrtSynchronizeStream(stream);
        if (wsp) aclrtFree(wsp);
        std::printf("NZ  B (format %d): ret=%d workspace=%.2f MB launch=%d\n",
                    fmt, r, ws / 1e6, r2);
        if (r2 == 0) {
            // Does it compute the same thing as the ND path?
            std::vector<uint16_t> a_out(static_cast<size_t>(N)), b_out(static_cast<size_t>(N));
            aclrtMemcpy(a_out.data(), a_out.size() * 2, o_nd, a_out.size() * 2,
                        ACL_MEMCPY_DEVICE_TO_HOST);
            aclrtMemcpy(b_out.data(), b_out.size() * 2, o_nz, b_out.size() * 2,
                        ACL_MEMCPY_DEVICE_TO_HOST);
            double max_diff = 0.0, mean_abs = 0.0;
            for (size_t i = 0; i < a_out.size(); ++i) {
                const double va = f16_to_f32(a_out[i]), vb = f16_to_f32(b_out[i]);
                mean_abs += std::fabs(va);
                max_diff = std::max(max_diff, std::fabs(va - vb));
            }
            mean_abs /= static_cast<double>(a_out.size());
            const bool match = max_diff < 0.02 * std::max(1e-3, mean_abs);
            std::printf("     vs ND: mean|ref|=%.4f max_diff=%.4f -> %s\n",
                        mean_abs, max_diff, match ? "MATCH" : "MISMATCH (layout wrong)");
            if (match && good_format < 0) good_format = fmt;
        }
        aclDestroyTensor(tb);
    }

    if (good_format < 0) {
        std::printf("\n-> No NZ format both accepted and numerically correct.\n");
        std::printf("   aclnnMm will not take a pre-formatted weight on this build.\n");
        return 1;
    }
    std::printf("\nusing format %d for the timed comparison\n\n", good_format);

    // Cold-read timing: distinct allocations, ND vs NZ.
    std::vector<void*> nd_bufs, nz_bufs;
    for (int i = 0; i < n_bufs; ++i) {
        void* p = nullptr;
        if (aclrtMalloc(&p, wbytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0) break;
        aclrtMemcpy(p, wbytes, nd_host.data(), wbytes, ACL_MEMCPY_HOST_TO_DEVICE);
        nd_bufs.push_back(p);
        void* q = nullptr;
        if (aclrtMalloc(&q, wbytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0) break;
        aclrtMemcpy(q, wbytes, nz_host.data(), wbytes, ACL_MEMCPY_HOST_TO_DEVICE);
        nz_bufs.push_back(q);
    }
    const int m = static_cast<int>(std::min(nd_bufs.size(), nz_bufs.size()));
    std::printf("cycling %d distinct buffers per layout\n\n", m);
    if (m < 2) return 1;

    auto cycle = [&](bool nz) {
        auto once = [&](int i) {
            aclTensor* tb = nz ? make_nz(nz_bufs[static_cast<size_t>(i)], K, N, good_format)
                               : make_nd(nd_bufs[static_cast<size_t>(i)], {K, N});
            uint64_t ws = 0;
            aclOpExecutor* ex = nullptr;
            if (aclnnMmGetWorkspaceSize(ta, tb, nz ? tc_nz : tc_nd,
                                        kCubeMathType, &ws, &ex) == 0) {
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
        return (now_ms() - t0) / m;
    };

    double nd_ms = cycle(false);
    double nz_ms = cycle(true);
    std::printf("ND weight : %8.4f ms -> %6.2f GB/s\n", nd_ms, wbytes / (nd_ms * 1e6));
    std::printf("NZ weight : %8.4f ms -> %6.2f GB/s\n", nz_ms, wbytes / (nz_ms * 1e6));
    std::printf("\n=== VERDICT ===\n");
    std::printf("ND=%.2f  NZ=%.2f ms  (%.2fx)  floor=%.2f ms\n",
                nd_ms, nz_ms, nd_ms / nz_ms, wbytes / 22.1e6);
    if (nz_ms < nd_ms * 0.8) {
        std::printf("-> WIN. Pre-format weights to NZ once at load time; the MLP's\n");
        std::printf("   3 projections/layer would drop from %.1f to %.1f ms.\n",
                    3 * nd_ms, 3 * nz_ms);
    } else {
        std::printf("-> No win from pre-formatting.\n");
    }

    for (void* p : nd_bufs) aclrtFree(p);
    for (void* p : nz_bufs) aclrtFree(p);
    return 0;
}
