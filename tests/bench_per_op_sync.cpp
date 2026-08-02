// How much of a small op's cost is the per-op device sync?
//
// run_op() calls aclrtSynchronizeStream after every single op, because it frees
// the op's workspace immediately afterwards. Decode issues thousands of tiny
// ops per token -- rope, rms_norms, the four attention ops -- and the profile
// shows them costing hundreds of microseconds each on tensors of a few KB,
// which is far more than the arithmetic. This measures the sync's share
// directly: identical op sequences, synced per op vs synced once at the end.
#include "minicpmo/acl_context.h"
#include "minicpmo/ops.h"
#include "minicpmo/tensor.h"

#include <aclnnop/aclnn_mul.h>

#include <chrono>
#include <cstdio>
#include <vector>

using namespace minicpmo;

namespace {

double ms_since(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
}

}  // namespace

int main() {
    AclContext ctx(0);
    aclrtStream stream = ctx.stream();

    // Decode-sized elementwise work: a few KB, the shape the small ops see.
    const int64_t n = 4096;
    Tensor a({1, n}, DType::Float16); a.allocate();
    Tensor b({1, n}, DType::Float16); b.allocate();
    Tensor out({1, n}, DType::Float16); out.allocate();
    aclrtMemset(a.data(), a.size_bytes(), 0, a.size_bytes());
    aclrtMemset(b.data(), b.size_bytes(), 0, b.size_bytes());

    const int iters = 500;

    // Warm up so JIT compilation is not counted.
    for (int i = 0; i < 20; ++i) mul(a, b, out, stream);
    aclrtSynchronizeStream(stream);

    // Path A: the production path -- mul() goes through run_op, which syncs.
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) mul(a, b, out, stream);
    aclrtSynchronizeStream(stream);
    const double per_op_sync = ms_since(t0) / iters;

    // Path B: same op, enqueued without a sync in between, synced once at the
    // end. Built by hand because mul() always syncs.
    std::vector<aclTensor*> keep;
    auto make = [&](const Tensor& t) {
        std::vector<int64_t> dims = t.shape();
        std::vector<int64_t> strides(dims.size(), 1);
        for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        aclTensor* h = aclCreateTensor(dims.data(), dims.size(), ACL_FLOAT16,
                                       strides.data(), 0, ACL_FORMAT_ND,
                                       dims.data(), dims.size(), const_cast<void*>(t.data()));
        keep.push_back(h);
        return h;
    };
    aclTensor* ha = make(a);
    aclTensor* hb = make(b);
    aclTensor* ho = make(out);

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        uint64_t ws = 0;
        aclOpExecutor* ex = nullptr;
        aclnnMulGetWorkspaceSize(ha, hb, ho, &ws, &ex);
        void* wsp = nullptr;
        if (ws > 0) aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnMul(wsp, ws, ex, stream);
        if (wsp) aclrtFree(wsp);  // safe only because ws is 0 for this op
    }
    aclrtSynchronizeStream(stream);
    const double batched = ms_since(t0) / iters;

    // Path C: how long the sync alone takes on an idle stream.
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) aclrtSynchronizeStream(stream);
    const double bare_sync = ms_since(t0) / iters;

    for (aclTensor* h : keep) aclDestroyTensor(h);

    std::printf("per-op sync (production path) : %8.3f ms/op\n", per_op_sync);
    std::printf("enqueue only, one sync at end : %8.3f ms/op\n", batched);
    std::printf("bare sync on idle stream      : %8.3f ms/op\n", bare_sync);
    std::printf("\nsync overhead share: %.1f%%  (%.3f ms of %.3f ms)\n",
                100.0 * (per_op_sync - batched) / per_op_sync,
                per_op_sync - batched, per_op_sync);
    return 0;
}
