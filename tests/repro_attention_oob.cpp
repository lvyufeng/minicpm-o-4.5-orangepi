// Repro for the aicore exception (507015) that the batched incre_flash_attention
// hits in the real model but never in a bare unit test.
//
// Hypothesis: the batched path runs BatchMatMul with M = group_size = 4. The
// Cube unit works in 16x16 fractal tiles, so it reads a full 16-row tile even
// when M is 4 -- 12 rows past the logical end. For every batch but the last
// those rows land on the next batch's data; for the last batch they land past
// the end of the buffer. In an almost-empty test process that memory is mapped
// and the read is harmless (the Cube masks the extra rows out of the result,
// which is why correctness tests pass). In the real model, with 10.4 GB
// resident, the page after the buffer can be unmapped and the read faults.
//
// This binary fills device memory first, then runs the same call. Set
// MINICPM_REPRO_FILL_GB to change how much is reserved (default 8).
#include "minicpmo/acl_context.h"
#include "minicpmo/ops.h"
#include "minicpmo/tensor.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace minicpmo;

int main() {
    AclContext ctx(0);
    aclrtStream stream = ctx.stream();

    const char* gb_env = std::getenv("MINICPM_REPRO_FILL_GB");
    const int fill_gb = gb_env ? std::atoi(gb_env) : 8;

    std::vector<void*> blocks;
    size_t filled = 0;
    const size_t block = 256ull * 1024 * 1024;
    for (int i = 0; i < fill_gb * 4; ++i) {
        void* p = nullptr;
        if (aclrtMalloc(&p, block, ACL_MEM_MALLOC_HUGE_FIRST) != 0) break;
        blocks.push_back(p);
        filled += block;
    }
    std::printf("Reserved %.2f GB of device memory in %zu blocks\n",
                filled / (1024.0 * 1024.0 * 1024.0), blocks.size());

    const int64_t num_q_heads = 32, num_kv_heads = 8, head_dim = 128;
    const int64_t max_seq = 4096;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    Tensor q({num_q_heads, head_dim}, DType::Float16); q.allocate();
    Tensor k({num_kv_heads, max_seq, head_dim}, DType::Float16); k.allocate();
    Tensor v({num_kv_heads, max_seq, head_dim}, DType::Float16); v.allocate();
    Tensor out({1, num_q_heads * head_dim}, DType::Float16); out.allocate();
    aclrtMemset(q.data(), q.size_bytes(), 0, q.size_bytes());
    aclrtMemset(k.data(), k.size_bytes(), 0, k.size_bytes());
    aclrtMemset(v.data(), v.size_bytes(), 0, v.size_bytes());

    // Walk the contexts a real generation sees, many times over, churning a
    // large workspace-sized allocation in between. The fault is intermittent in
    // the real model, which points at the address space rather than the shapes:
    // whether an over-read lands on a mapped page depends on what the allocator
    // last handed back. Static fill alone did not reproduce it; churn might.
    const char* iter_env = std::getenv("MINICPM_REPRO_ITERS");
    const int iters = iter_env ? std::atoi(iter_env) : 200;
    const int64_t contexts[] = {1, 15, 16, 17, 31, 32, 47, 64, 100};

    for (int it = 0; it < iters; ++it) {
        for (int64_t context : contexts) {
            try {
                incre_flash_attention(q, k, v, context, num_q_heads, num_kv_heads,
                                      head_dim, scale, out, stream);
            } catch (const std::exception& e) {
                std::printf("FAULT at iter=%d context=%ld: %s\n",
                            it, static_cast<long>(context), e.what());
                for (void* p : blocks) aclrtFree(p);
                return 1;
            }
        }
        // Churn: an MLP-sized transient, freed immediately, so the next
        // attention call sees a different allocator state.
        void* churn = nullptr;
        if (aclrtMalloc(&churn, 101ull * 1024 * 1024, ACL_MEM_MALLOC_HUGE_FIRST) == 0) {
            aclrtFree(churn);
        }
        if ((it + 1) % 20 == 0) {
            std::printf("  %d/%d iterations clean\n", it + 1, iters);
            std::fflush(stdout);
        }
    }

    for (void* p : blocks) aclrtFree(p);
    std::printf("\nNo fault reproduced: %d iterations x %zu contexts at %d GB fill\n",
                iters, sizeof(contexts) / sizeof(contexts[0]), fill_gb);
    return 0;
}
