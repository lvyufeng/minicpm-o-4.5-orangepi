#include "minicpmo/profiling.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace minicpmo {
namespace {

struct Entry {
    double total_ms = 0.0;
    int64_t count = 0;
};

std::mutex g_mutex;
std::unordered_map<std::string, Entry> g_entries;

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

bool profiling_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("MINICPM_PROFILE");
        return v != nullptr && std::string(v) != "0" && std::string(v) != "false";
    }();
    return enabled;
}

void profile_add(const char* category, double ms) {
    std::lock_guard<std::mutex> lock(g_mutex);
    Entry& e = g_entries[category];
    e.total_ms += ms;
    e.count += 1;
}

void profile_reset() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_entries.clear();
}

void profile_print() {
    std::vector<std::pair<std::string, Entry>> rows;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        rows.assign(g_entries.begin(), g_entries.end());
    }
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        return a.second.total_ms > b.second.total_ms;
    });
    std::cerr << "\n[Profile] category               total_ms      count      avg_ms\n";
    std::cerr << "----------------------------------------------------------------\n";
    for (const auto& [name, e] : rows) {
        std::cerr << "[Profile] " << name;
        for (size_t i = name.size(); i < 24; ++i) std::cerr << ' ';
        std::cerr << " " << e.total_ms << "  " << e.count << "  "
                  << (e.count > 0 ? e.total_ms / static_cast<double>(e.count) : 0.0) << "\n";
    }
    std::cerr << std::endl;
}

ProfileScope::ProfileScope(const char* category, aclrtStream stream)
    : category_(category), stream_(stream), active_(profiling_enabled()), start_ms_(0.0) {
    if (active_) {
        aclrtSynchronizeStream(stream_);
        start_ms_ = now_ms();
    }
}

ProfileScope::~ProfileScope() {
    if (active_) {
        aclrtSynchronizeStream(stream_);
        profile_add(category_, now_ms() - start_ms_);
    }
}

}  // namespace minicpmo
