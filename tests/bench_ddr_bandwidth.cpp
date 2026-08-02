// Standalone microbenchmark: measures raw device-to-device memcpy bandwidth
// via aclrtMemcpyAsync, to establish the hardware ceiling that
// bench_mlp_matmul's weight-streaming GB/s figures should be compared
// against. If matmul's effective GB/s is far below this ceiling, the
// bottleneck is compute/kernel-launch overhead, not memory bandwidth.
#include "minicpmo/acl_context.h"
#include "minicpmo/tensor.h"

#include <chrono>
#include <cstdio>

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

    struct Size { size_t bytes; const char* label; };
    std::vector<Size> sizes = {
        {4ull * 1024 * 1024, "4 MB"},
        {32ull * 1024 * 1024, "32 MB"},
        {100ull * 1024 * 1024, "100 MB"},
    };

    std::printf("%-10s %10s %12s\n", "size", "avg_ms", "GB/s");
    std::printf("----------------------------------\n");
    for (const auto& s : sizes) {
        Tensor src({static_cast<int64_t>(s.bytes)}, DType::UInt8); src.allocate();
        Tensor dst({static_cast<int64_t>(s.bytes)}, DType::UInt8); dst.allocate();

        const int warmup = 5, iters = 20;
        for (int i = 0; i < warmup; ++i) {
            check_acl(aclrtMemcpyAsync(dst.data(), s.bytes, src.data(), s.bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream), "warmup memcpy");
        }
        check_acl(aclrtSynchronizeStream(stream), "warmup sync");

        double t0 = now_ms();
        for (int i = 0; i < iters; ++i) {
            check_acl(aclrtMemcpyAsync(dst.data(), s.bytes, src.data(), s.bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream), "timed memcpy");
        }
        check_acl(aclrtSynchronizeStream(stream), "timed sync");
        double t1 = now_ms();

        double avg_ms = (t1 - t0) / iters;
        // D2D copy touches the byte range twice (read src + write dst).
        double gb_per_s = (2.0 * static_cast<double>(s.bytes)) / (avg_ms * 1e-3) / 1e9;
        std::printf("%-10s %10.4f %12.2f\n", s.label, avg_ms, gb_per_s);
    }

    return 0;
}
