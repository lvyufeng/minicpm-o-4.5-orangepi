// Correctness guard for the batched incre_flash_attention.
//
// The decode attention path is hand-rolled: there is no IncreFlashAttention
// kernel for 310B, so it is built from BatchMatMul + Muls + Add + Softmax over
// GQA-shaped strided views. Three things in it can be silently wrong rather
// than loud: the [B, G, D] view of a flat [num_q_heads * head_dim] query, the
// stride-trick transpose feeding q @ k^T, and the stride-0 broadcast of the
// pad mask. Any of those produces plausible numbers and garbage generation,
// so check against a CPU fp32 reference on the real decode shapes plus the
// bucket boundaries where padding actually bites.
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

// Cache is [num_kv_heads, max_seq, head_dim]; only the first `context` rows of
// each plane hold real data.
void check(int64_t context, int64_t max_seq, int64_t num_q_heads,
           int64_t num_kv_heads, int64_t head_dim, aclrtStream stream) {
    const int64_t group_size = num_q_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::mt19937 rng(static_cast<unsigned>(context * 7919 + max_seq * 31 + num_q_heads));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<uint16_t> qh(static_cast<size_t>(num_q_heads * head_dim));
    std::vector<float> qf(qh.size());
    for (size_t i = 0; i < qh.size(); ++i) {
        qh[i] = f32_to_f16(dist(rng));
        qf[i] = f16_to_f32(qh[i]);
    }

    const size_t cache_n = static_cast<size_t>(num_kv_heads * max_seq * head_dim);
    std::vector<uint16_t> kh(cache_n, 0), vh(cache_n, 0);
    std::vector<float> kf(cache_n, 0.0f), vf(cache_n, 0.0f);
    for (int64_t h = 0; h < num_kv_heads; ++h) {
        for (int64_t t = 0; t < context; ++t) {
            for (int64_t d = 0; d < head_dim; ++d) {
                const size_t i = static_cast<size_t>((h * max_seq + t) * head_dim + d);
                kh[i] = f32_to_f16(dist(rng));
                vh[i] = f32_to_f16(dist(rng));
                kf[i] = f16_to_f32(kh[i]);
                vf[i] = f16_to_f32(vh[i]);
            }
        }
    }
    // Poison the padded tail: a correct mask makes these unreachable. If the
    // mask or the broadcast is wrong, large garbage here blows up the result
    // instead of shifting it slightly.
    for (int64_t h = 0; h < num_kv_heads; ++h) {
        for (int64_t t = context; t < max_seq; ++t) {
            for (int64_t d = 0; d < head_dim; ++d) {
                const size_t i = static_cast<size_t>((h * max_seq + t) * head_dim + d);
                kh[i] = f32_to_f16(50.0f);
                vh[i] = f32_to_f16(-100.0f);
            }
        }
    }

    Tensor q({1, num_q_heads * head_dim}, DType::Float16); q.allocate();
    q.copy_from_host(qh.data(), qh.size() * sizeof(uint16_t));
    Tensor k({num_kv_heads, max_seq, head_dim}, DType::Float16); k.allocate();
    k.copy_from_host(kh.data(), kh.size() * sizeof(uint16_t));
    Tensor v({num_kv_heads, max_seq, head_dim}, DType::Float16); v.allocate();
    v.copy_from_host(vh.data(), vh.size() * sizeof(uint16_t));
    Tensor out({1, num_q_heads * head_dim}, DType::Float16); out.allocate();

    incre_flash_attention(q, k, v, context, num_q_heads, num_kv_heads, head_dim,
                          scale, out, stream);
    aclrtSynchronizeStream(stream);

    std::vector<uint16_t> got(static_cast<size_t>(num_q_heads * head_dim));
    out.copy_to_host(got.data(), got.size() * sizeof(uint16_t));

    double max_abs_err = 0.0, mean_abs_ref = 0.0;
    int n_nonfinite = 0;
    for (int64_t qh_i = 0; qh_i < num_q_heads; ++qh_i) {
        const int64_t kv_h = qh_i / group_size;

        std::vector<double> s(static_cast<size_t>(context));
        double smax = -1e30;
        for (int64_t t = 0; t < context; ++t) {
            double dot = 0.0;
            for (int64_t d = 0; d < head_dim; ++d) {
                dot += static_cast<double>(qf[static_cast<size_t>(qh_i * head_dim + d)]) *
                       kf[static_cast<size_t>((kv_h * max_seq + t) * head_dim + d)];
            }
            s[static_cast<size_t>(t)] = dot * scale;
            if (s[static_cast<size_t>(t)] > smax) smax = s[static_cast<size_t>(t)];
        }
        double sum = 0.0;
        for (int64_t t = 0; t < context; ++t) {
            s[static_cast<size_t>(t)] = std::exp(s[static_cast<size_t>(t)] - smax);
            sum += s[static_cast<size_t>(t)];
        }
        for (int64_t d = 0; d < head_dim; ++d) {
            double ref = 0.0;
            for (int64_t t = 0; t < context; ++t) {
                ref += s[static_cast<size_t>(t)] *
                       vf[static_cast<size_t>((kv_h * max_seq + t) * head_dim + d)];
            }
            ref /= sum;
            const double g = f16_to_f32(got[static_cast<size_t>(qh_i * head_dim + d)]);
            if (!std::isfinite(g)) { ++n_nonfinite; continue; }
            mean_abs_ref += std::fabs(ref);
            const double e = std::fabs(g - ref);
            if (e > max_abs_err) max_abs_err = e;
        }
    }
    mean_abs_ref /= static_cast<double>(num_q_heads * head_dim);

    // Output is a convex combination of V values, so it lives on V's scale.
    // Softmax in fp16 loses a few ulps per term; judge on absolute error
    // against the typical magnitude rather than relative error near zero.
    const double tol = std::max(0.02, 0.05 * mean_abs_ref);
    const bool ok = (max_abs_err <= tol) && (n_nonfinite == 0);
    std::printf("ctx=%-5ld max_seq=%-5ld qh=%-3ld kvh=%-3ld d=%-4ld "
                "mean|ref|=%7.4f max_abs_err=%7.4f tol=%6.4f nonfinite=%-4d %s\n",
                static_cast<long>(context), static_cast<long>(max_seq),
                static_cast<long>(num_q_heads), static_cast<long>(num_kv_heads),
                static_cast<long>(head_dim), mean_abs_ref, max_abs_err, tol,
                n_nonfinite, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

}  // namespace

int main() {
    AclContext ctx(0);
    aclrtStream stream = ctx.stream();

    // Small shapes first: a failure here is easier to read than at 32 heads.
    check(4, 64, 4, 2, 64, stream);
    check(16, 64, 4, 2, 64, stream);

    // Real decode geometry: 32 query heads, 8 KV heads, head_dim 128.
    std::printf("\n--- production shape (32q/8kv/128d) ---\n");
    check(1, 4096, 32, 8, 128, stream);      // first decode step
    check(15, 4096, 32, 8, 128, stream);     // just after a short prefill
    check(16, 4096, 32, 8, 128, stream);     // exactly on a bucket boundary
    check(17, 4096, 32, 8, 128, stream);     // one past a boundary: max padding
    check(100, 4096, 32, 8, 128, stream);    // mid-bucket, not a multiple of 16
    check(128, 4096, 32, 8, 128, stream);    // bucket boundary, no padding
    check(1000, 4096, 32, 8, 128, stream);   // long context

    std::printf("\n%s (%d failing cases)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
