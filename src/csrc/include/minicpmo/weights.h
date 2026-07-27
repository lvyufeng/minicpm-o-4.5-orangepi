#pragma once

#include "minicpmo/tensor.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace minicpmo {

struct TensorInfo {
    DType dtype;
    std::vector<int64_t> shape;
    uint64_t data_begin;
    uint64_t data_end;
    std::string shard_file;  // For sharded models, which file contains this tensor
};

class WeightsIndex {
public:
    explicit WeightsIndex(const std::string& safetensors_path);
    ~WeightsIndex();

    const TensorInfo& at(const std::string& name) const;
    bool contains(const std::string& name) const;
    size_t size() const { return tensors_.size(); }
    std::vector<std::string> names() const;

    Tensor load_to_device(const std::string& name) const;
    Tensor load_to_device_as(const std::string& name, DType target_dtype) const;

    // Release shard file memory after all weights are loaded (no-op in streaming mode)
    void release_shard_memory();

private:
    std::string path_;
    std::string model_dir_;  // Directory containing shard files
    std::vector<uint8_t> file_bytes_;  // Only for single-file models
    std::unordered_map<std::string, TensorInfo> tensors_;

    void parse();
    void parse_sharded_index(const std::string& index_path);
    void parse_shard_header(const std::string& shard_path);

    // Load tensor data directly from file without caching
    void load_tensor_data(const std::string& shard_file, uint64_t offset, uint64_t size, void* buffer) const;
};

// Resolve the path to the MiniCPM-V 4.6 safetensors file used by tests and
// benches. Honors the MINICPMV_MODEL_PATH environment variable; falls back to
// `./MiniCPM-V-4.6/model.safetensors` relative to the binary's CWD.
std::string default_safetensors_path();

}  // namespace minicpmo
