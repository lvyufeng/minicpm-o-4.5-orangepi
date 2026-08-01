// Correctness guard for every B layout matmul_b_transposed accepts.
//
// Decode weights are packed into FRACTAL_NZ at load time so aclnnMm consumes
// them without its per-call 101 MB reformat; that layout is hand-built (see
// pack_fractal_nz), so a wrong permutation would still "run" and silently
// compute garbage. Every decode projection goes through here, so a slip
// corrupts all generation. Checks against a CPU fp32 reference on the real
// decode shapes, for [K,N] natural, [N,K] legacy, and FRACTAL_NZ.
#include "minicpmo/acl_context.h"
#include "minicpmo/ops.h"
#include "minicpmo/tensor.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace minicpmo;

namespace {

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
    if (exp == 0) {
        out = sign;
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

int failures = 0;

enum class BLayout { Natural, Legacy, Nz };

const char* layout_name(BLayout l) {
    switch (l) {
        case BLayout::Natural: return "[K,N]";
        case BLayout::Legacy:  return "[N,K]";
        case BLayout::Nz:      return "NZ";
    }
    return "?";
}

// Natural -> B stored [K, N], b[k*N + n]
// Legacy  -> B stored [N, K], b[n*K + k]
// Nz      -> B logically [K, N], stored FRACTAL_NZ
void check(int64_t K, int64_t N, BLayout layout, aclrtStream stream) {
    const bool b_natural = (layout != BLayout::Legacy);
    std::mt19937 rng(static_cast<unsigned>(K * 131 + N * 17 + (b_natural ? 1 : 0)));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<uint16_t> xh(static_cast<size_t>(K));
    std::vector<float> xf(static_cast<size_t>(K));
    for (int64_t i = 0; i < K; ++i) {
        xf[static_cast<size_t>(i)] = dist(rng);
        xh[static_cast<size_t>(i)] = f32_to_f16(xf[static_cast<size_t>(i)]);
    }

    const size_t wn = static_cast<size_t>(K) * static_cast<size_t>(N);
    std::vector<uint16_t> wh(wn);
    std::vector<float> wf(wn);
    for (size_t i = 0; i < wn; ++i) {
        wf[i] = dist(rng) * 0.05f;
        wh[i] = f32_to_f16(wf[i]);
        wf[i] = f16_to_f32(wh[i]);  // reference uses the rounded values
    }

    Tensor x({1, K}, DType::Float16); x.allocate();
    x.copy_from_host(xh.data(), xh.size() * sizeof(uint16_t));
    Tensor w(b_natural ? std::vector<int64_t>{K, N} : std::vector<int64_t>{N, K},
             DType::Float16);
    w.allocate();
    if (layout == BLayout::Nz) {
        std::vector<uint16_t> packed(wn);
        pack_fractal_nz(wh.data(), packed.data(), K, N);
        w.copy_from_host(packed.data(), packed.size() * sizeof(uint16_t));
        w.set_format(Format::FractalNz);
    } else {
        w.copy_from_host(wh.data(), wh.size() * sizeof(uint16_t));
    }
    Tensor out({1, N}, DType::Float16); out.allocate();

    matmul_b_transposed(x, w, out, stream);
    aclrtSynchronizeStream(stream);

    std::vector<uint16_t> got(static_cast<size_t>(N));
    out.copy_to_host(got.data(), got.size() * sizeof(uint16_t));

    double max_rel = 0.0, max_abs_err = 0.0, mean_abs_ref = 0.0;
    int n_bad = 0;
    for (int64_t n = 0; n < N; ++n) {
        double ref = 0.0;
        for (int64_t k = 0; k < K; ++k) {
            const size_t wi = b_natural
                ? static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(n)
                : static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k);
            ref += static_cast<double>(xf[static_cast<size_t>(k)]) * wf[wi];
        }
        const double g = f16_to_f32(got[static_cast<size_t>(n)]);
        if (!std::isfinite(g)) {
            ++n_bad;
            continue;
        }
        mean_abs_ref += std::fabs(ref);
        const double abs_err = std::fabs(g - ref);
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        const double denom = std::max(1e-3, std::fabs(ref));
        const double rel = abs_err / denom;
        if (rel > max_rel) max_rel = rel;
        if (rel > 0.05) ++n_bad;
    }
    mean_abs_ref /= static_cast<double>(N);

    // fp16 has ~3 decimal digits; a K-long dot product accumulates rounding
    // proportional to sqrt(K) * eps * |terms|. Judge on absolute error scaled
    // by the typical output magnitude, not on relative error at near-zero
    // outputs where fp16 has no precision to give.
    const double tol = 8.0 * std::sqrt(static_cast<double>(K)) * 4.9e-4 * mean_abs_ref;
    const bool ok = (max_abs_err <= tol);
    std::printf("K=%-6ld N=%-6ld B=%-8s mean|ref|=%8.4f max_abs_err=%8.4f tol=%8.4f "
                "max_rel=%9.4f rel>5%%=%-5d %s\n",
                static_cast<long>(K), static_cast<long>(N),
                layout_name(layout), mean_abs_ref, max_abs_err, tol,
                max_rel, n_bad, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

}  // namespace

int main() {
    AclContext ctx(0);
    aclrtStream stream = ctx.stream();

    // Real decode shapes: MLP gate/up (4096->12288), MLP down (12288->4096),
    // plus small shapes and a non-multiple-of-16 N to exercise the guard.
    check(256, 512, BLayout::Natural, stream);
    check(256, 512, BLayout::Legacy, stream);
    check(4096, 12288, BLayout::Natural, stream);
    check(12288, 4096, BLayout::Natural, stream);
    // Note: N == K is ambiguous by matmul_b_transposed's shape-based
    // convention (bIsTransposed wins), so a [K,N]-stored square weight is
    // interpreted as [N,K]. NZ carries an explicit format flag and is exempt.
    check(4096, 4096, BLayout::Legacy, stream);
    check(512, 100, BLayout::Natural, stream);  // N not a multiple of 16

    // FRACTAL_NZ weights: the production decode layout.
    std::printf("\n--- FRACTAL_NZ ---\n");
    check(256, 512, BLayout::Nz, stream);
    check(4096, 12288, BLayout::Nz, stream);
    check(12288, 4096, BLayout::Nz, stream);
    check(4096, 4096, BLayout::Nz, stream);   // square: only valid via NZ

    std::printf("\n%s (%d failing shapes)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
