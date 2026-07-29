#pragma once

// Minimal opt-in wall-clock profiling for the decode hot path.
//
// Enabled by setting the environment variable MINICPM_PROFILE=1 before
// starting the process. When disabled (the default), profile_scope() has
// negligible overhead (one cached env lookup, no stream sync) and does not
// affect production performance.
//
// Usage:
//   {
//       ProfileScope _p("attention", stream);
//       incre_flash_attention(...);
//   }  // records elapsed device execution time under "attention" on destruction
//
// Because ACL ops are enqueued asynchronously, ProfileScope synchronizes the
// stream at both the start and end of its lifetime when profiling is
// enabled, so the reported time reflects real device execution time for that
// scope (not just host-side enqueue time). This makes profiling intrusive on
// the pipeline -- only enable it for diagnostic runs, not production serving.

#include <acl/acl.h>

namespace minicpmo {

bool profiling_enabled();

// Accumulate `ms` milliseconds under `category`, incrementing its hit count.
void profile_add(const char* category, double ms);

// Clear all accumulated profiling data.
void profile_reset();

// Print a summary (category, total ms, hit count, avg ms) to stderr, sorted
// by descending total time.
void profile_print();

class ProfileScope {
public:
    ProfileScope(const char* category, aclrtStream stream);
    ~ProfileScope();

    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;

private:
    const char* category_;
    aclrtStream stream_;
    bool active_;
    double start_ms_;
};

}  // namespace minicpmo
