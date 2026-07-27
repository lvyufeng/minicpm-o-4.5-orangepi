#include "minicpmo/weights.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace minicpmo {
namespace {

DType parse_dtype(const std::string& s) {
    if (s == "F16") return DType::Float16;
    if (s == "BF16") return DType::BFloat16;
    if (s == "F32") return DType::Float32;
    if (s == "I32") return DType::Int32;
    if (s == "I64") return DType::Int64;
    if (s == "U8") return DType::UInt8;
    throw std::runtime_error("unsupported safetensors dtype: " + s);
}

uint16_t bf16_bits_to_f16_bits(uint16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
        ++i;
    }
}

void expect(const std::string& s, size_t& i, char ch) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != ch) {
        std::ostringstream oss;
        oss << "expected '" << ch << "' at position " << i;
        throw std::runtime_error(oss.str());
    }
    ++i;
}

std::string parse_string(const std::string& s, size_t& i) {
    skip_ws(s, i);
    expect(s, i, '"');
    std::string out;
    while (i < s.size()) {
        char c = s[i++];
        if (c == '"') {
            return out;
        }
        if (c == '\\') {
            if (i >= s.size()) throw std::runtime_error("bad escape in json string");
            char esc = s[i++];
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default: throw std::runtime_error("unsupported json escape");
            }
        } else {
            out.push_back(c);
        }
    }
    throw std::runtime_error("unterminated json string");
}

uint64_t parse_uint(const std::string& s, size_t& i) {
    skip_ws(s, i);
    if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) {
        throw std::runtime_error("expected unsigned integer");
    }
    uint64_t value = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
        value = value * 10 + static_cast<uint64_t>(s[i] - '0');
        ++i;
    }
    return value;
}

std::vector<int64_t> parse_int64_array(const std::string& s, size_t& i) {
    std::vector<int64_t> out;
    expect(s, i, '[');
    skip_ws(s, i);
    if (i < s.size() && s[i] == ']') {
        ++i;
        return out;
    }
    while (true) {
        out.push_back(static_cast<int64_t>(parse_uint(s, i)));
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
            ++i;
            continue;
        }
        break;
    }
    expect(s, i, ']');
    return out;
}

std::vector<uint64_t> parse_uint64_array(const std::string& s, size_t& i) {
    std::vector<uint64_t> out;
    expect(s, i, '[');
    skip_ws(s, i);
    if (i < s.size() && s[i] == ']') {
        ++i;
        return out;
    }
    while (true) {
        out.push_back(parse_uint(s, i));
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
            ++i;
            continue;
        }
        break;
    }
    expect(s, i, ']');
    return out;
}

void skip_json_value(const std::string& s, size_t& i);

void skip_json_object(const std::string& s, size_t& i) {
    expect(s, i, '{');
    skip_ws(s, i);
    if (i < s.size() && s[i] == '}') {
        ++i;
        return;
    }
    while (true) {
        (void)parse_string(s, i);
        expect(s, i, ':');
        skip_json_value(s, i);
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
            ++i;
            continue;
        }
        break;
    }
    expect(s, i, '}');
}

void skip_json_array(const std::string& s, size_t& i) {
    expect(s, i, '[');
    skip_ws(s, i);
    if (i < s.size() && s[i] == ']') {
        ++i;
        return;
    }
    while (true) {
        skip_json_value(s, i);
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
            ++i;
            continue;
        }
        break;
    }
    expect(s, i, ']');
}

void skip_json_value(const std::string& s, size_t& i) {
    skip_ws(s, i);
    if (i >= s.size()) throw std::runtime_error("unexpected end of json");
    char c = s[i];
    if (c == '{') {
        skip_json_object(s, i);
    } else if (c == '[') {
        skip_json_array(s, i);
    } else if (c == '"') {
        (void)parse_string(s, i);
    } else {
        while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') {
            ++i;
        }
    }
}

}  // namespace

WeightsIndex::WeightsIndex(const std::string& safetensors_path) : path_(safetensors_path) {
    // Extract directory from path
    size_t last_slash = safetensors_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        model_dir_ = safetensors_path.substr(0, last_slash + 1);
    } else {
        model_dir_ = "";
    }

    // Check if the main file exists
    std::ifstream main_file(safetensors_path);

    if (main_file.good()) {
        // Single file model - load directly into memory (smaller models)
        main_file.seekg(0, std::ios::end);
        auto size = static_cast<size_t>(main_file.tellg());
        main_file.seekg(0, std::ios::beg);
        file_bytes_.resize(size);
        main_file.read(reinterpret_cast<char*>(file_bytes_.data()), static_cast<std::streamsize>(size));
        parse();
    } else {
        // Try sharded model - check for .index.json
        std::string index_path = safetensors_path + ".index.json";
        std::ifstream index_file(index_path);

        if (index_file.good()) {
            // Sharded model - parse index file and shard headers only
            std::cout << "[WeightsIndex] Loading sharded model (streaming mode)" << std::endl;
            parse_sharded_index(index_path);
        } else {
            throw std::runtime_error("failed to open safetensors file: " + safetensors_path);
        }
    }
}

WeightsIndex::~WeightsIndex() {
    // No cleanup needed - we don't cache shard data anymore
}

const TensorInfo& WeightsIndex::at(const std::string& name) const {
    return tensors_.at(name);
}

bool WeightsIndex::contains(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

std::vector<std::string> WeightsIndex::names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& kv : tensors_) {
        out.push_back(kv.first);
    }
    std::sort(out.begin(), out.end());
    return out;
}

Tensor WeightsIndex::load_to_device(const std::string& name) const {
    const auto& info = at(name);
    Tensor tensor(info.shape, info.dtype);
    tensor.allocate();

    if (!info.shard_file.empty()) {
        // Streaming load from shard file - allocate temporary buffer
        size_t data_size = info.data_end - info.data_begin;
        std::vector<uint8_t> buffer(data_size);
        load_tensor_data(info.shard_file, info.data_begin, data_size, buffer.data());
        tensor.copy_from_host(buffer.data(), data_size);
    } else {
        // Load from main file (already in memory for small models)
        tensor.copy_from_host(file_bytes_.data() + info.data_begin, info.data_end - info.data_begin);
    }

    return tensor;
}

Tensor WeightsIndex::load_to_device_as(const std::string& name, DType target_dtype) const {
    const auto& info = at(name);
    if (info.dtype == target_dtype) {
        return load_to_device(name);
    }
    if (!(info.dtype == DType::BFloat16 && target_dtype == DType::Float16)) {
        throw std::runtime_error("unsupported dtype conversion in load_to_device_as");
    }

    const size_t numel = [&]() {
        size_t n = 1;
        for (auto d : info.shape) n *= static_cast<size_t>(d);
        return n;
    }();

    // Load data from file
    size_t data_size = info.data_end - info.data_begin;
    std::vector<uint8_t> buffer(data_size);

    if (!info.shard_file.empty()) {
        load_tensor_data(info.shard_file, info.data_begin, data_size, buffer.data());
    } else {
        std::memcpy(buffer.data(), file_bytes_.data() + info.data_begin, data_size);
    }

    // Convert BF16 to F16
    std::vector<uint16_t> converted(numel);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(buffer.data());
    for (size_t i = 0; i < numel; ++i) {
        converted[i] = bf16_bits_to_f16_bits(src[i]);
    }

    Tensor tensor(info.shape, target_dtype);
    tensor.allocate();
    tensor.copy_from_host(converted.data(), converted.size() * sizeof(uint16_t));
    return tensor;
}

void WeightsIndex::parse() {
    if (file_bytes_.size() < 8) {
        throw std::runtime_error("invalid safetensors file");
    }
    uint64_t header_len = 0;
    std::memcpy(&header_len, file_bytes_.data(), sizeof(uint64_t));
    if (8 + header_len > file_bytes_.size()) {
        throw std::runtime_error("invalid safetensors header length");
    }
    std::string header(reinterpret_cast<const char*>(file_bytes_.data() + 8), static_cast<size_t>(header_len));
    uint64_t data_base = 8 + header_len;

    size_t i = 0;
    expect(header, i, '{');
    skip_ws(header, i);
    if (i < header.size() && header[i] == '}') {
        ++i;
        return;
    }

    while (true) {
        std::string tensor_name = parse_string(header, i);
        expect(header, i, ':');
        if (tensor_name == "__metadata__") {
            skip_json_object(header, i);
        } else {
            expect(header, i, '{');
            TensorInfo info{};
            bool have_dtype = false, have_shape = false, have_offsets = false;
            while (true) {
                std::string key = parse_string(header, i);
                expect(header, i, ':');
                if (key == "dtype") {
                    info.dtype = parse_dtype(parse_string(header, i));
                    have_dtype = true;
                } else if (key == "shape") {
                    info.shape = parse_int64_array(header, i);
                    have_shape = true;
                } else if (key == "data_offsets") {
                    auto offsets = parse_uint64_array(header, i);
                    if (offsets.size() != 2) {
                        throw std::runtime_error("data_offsets size must be 2");
                    }
                    info.data_begin = data_base + offsets[0];
                    info.data_end = data_base + offsets[1];
                    have_offsets = true;
                } else {
                    skip_json_value(header, i);
                }
                skip_ws(header, i);
                if (i < header.size() && header[i] == ',') {
                    ++i;
                    continue;
                }
                break;
            }
            expect(header, i, '}');
            if (!have_dtype || !have_shape || !have_offsets) {
                throw std::runtime_error("incomplete tensor entry for " + tensor_name);
            }
            tensors_.emplace(std::move(tensor_name), std::move(info));
        }
        skip_ws(header, i);
        if (i < header.size() && header[i] == ',') {
            ++i;
            continue;
        }
        break;
    }
    expect(header, i, '}');
}

std::string default_safetensors_path() {
    if (const char* env = std::getenv("MINICPMV_MODEL_PATH")) {
        if (*env) {
            return std::string(env) + "/model.safetensors";
        }
    }
    return "./MiniCPM-V-4.6/model.safetensors";
}

void WeightsIndex::parse_sharded_index(const std::string& index_path) {
    std::cout << "[WeightsIndex] Parsing sharded index: " << index_path << std::endl;
    // Read index file
    std::ifstream in(index_path);
    if (!in) {
        throw std::runtime_error("failed to open index file: " + index_path);
    }
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    // Parse JSON to get weight_map
    size_t i = 0;
    expect(content, i, '{');

    std::set<std::string> shard_files;

    while (true) {
        skip_ws(content, i);
        if (i < content.size() && content[i] == '}') {
            ++i;
            break;
        }

        std::string key = parse_string(content, i);
        expect(content, i, ':');

        if (key == "weight_map") {
            expect(content, i, '{');
            // Parse weight_map: { "tensor_name": "shard_file.safetensors", ... }
            while (true) {
                skip_ws(content, i);
                if (i < content.size() && content[i] == '}') {
                    ++i;
                    break;
                }

                std::string tensor_name = parse_string(content, i);
                expect(content, i, ':');
                std::string shard_file = parse_string(content, i);

                // Collect unique shard files
                shard_files.insert(shard_file);

                skip_ws(content, i);
                if (i < content.size() && content[i] == ',') {
                    ++i;
                    continue;
                }
                break;
            }
        } else {
            skip_json_value(content, i);
        }

        skip_ws(content, i);
        if (i < content.size() && content[i] == ',') {
            ++i;
            continue;
        }
        break;
    }

    // Parse headers of all shard files
    for (const auto& shard_file : shard_files) {
        std::cout << "[WeightsIndex] Parsing shard header: " << shard_file << std::endl;
        parse_shard_header(model_dir_ + shard_file);
    }

    std::cout << "[WeightsIndex] Loaded " << tensors_.size() << " tensors from "
              << shard_files.size() << " shards (streaming mode)" << std::endl;
}

void WeightsIndex::parse_shard_header(const std::string& shard_path) {
    // Extract just the filename for the map key
    std::string shard_name = shard_path;
    size_t last_slash = shard_name.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        shard_name = shard_name.substr(last_slash + 1);
    }

    std::ifstream in(shard_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open shard file: " + shard_path);
    }

    // Read only the header, not the entire file
    uint64_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), sizeof(uint64_t));
    if (!in) {
        throw std::runtime_error("failed to read header length from: " + shard_path);
    }

    if (header_len > 100 * 1024 * 1024) {  // Sanity check: header shouldn't be > 100MB
        throw std::runtime_error("invalid safetensors header length");
    }

    std::vector<char> header_data(header_len);
    in.read(header_data.data(), static_cast<std::streamsize>(header_len));
    if (!in) {
        throw std::runtime_error("failed to read header from: " + shard_path);
    }

    std::string header(header_data.data(), header_len);
    uint64_t data_base = 8 + header_len;

    // Parse header and add tensors
    size_t idx = 0;
    expect(header, idx, '{');
    skip_ws(header, idx);

    while (true) {
        if (idx < header.size() && header[idx] == '}') {
            ++idx;
            break;
        }

        std::string tensor_name = parse_string(header, idx);
        expect(header, idx, ':');

        if (tensor_name == "__metadata__") {
            skip_json_object(header, idx);
        } else {
            expect(header, idx, '{');
            TensorInfo info{};
            info.shard_file = shard_name;
            bool have_dtype = false, have_shape = false, have_offsets = false;

            while (true) {
                std::string key = parse_string(header, idx);
                expect(header, idx, ':');

                if (key == "dtype") {
                    info.dtype = parse_dtype(parse_string(header, idx));
                    have_dtype = true;
                } else if (key == "shape") {
                    info.shape = parse_int64_array(header, idx);
                    have_shape = true;
                } else if (key == "data_offsets") {
                    auto offsets = parse_uint64_array(header, idx);
                    if (offsets.size() != 2) {
                        throw std::runtime_error("data_offsets size must be 2");
                    }
                    info.data_begin = data_base + offsets[0];
                    info.data_end = data_base + offsets[1];
                    have_offsets = true;
                } else {
                    skip_json_value(header, idx);
                }

                skip_ws(header, idx);
                if (idx < header.size() && header[idx] == ',') {
                    ++idx;
                    continue;
                }
                break;
            }
            expect(header, idx, '}');

            if (!have_dtype || !have_shape || !have_offsets) {
                throw std::runtime_error("incomplete tensor entry for " + tensor_name);
            }

            tensors_.emplace(std::move(tensor_name), std::move(info));
        }

        skip_ws(header, idx);
        if (idx < header.size() && header[idx] == ',') {
            ++idx;
            continue;
        }
        break;
    }
}

void WeightsIndex::release_shard_memory() {
    // In streaming mode, we don't cache shard data, so nothing to release
    // Only clear single-file model data if present
    if (!file_bytes_.empty()) {
        std::cout << "[WeightsIndex] Releasing single-file model memory..." << std::endl;
        size_t bytes = file_bytes_.size();
        file_bytes_.clear();
        file_bytes_.shrink_to_fit();
        std::cout << "[WeightsIndex] Released " << (bytes / 1024 / 1024) << " MB of host memory" << std::endl;
    }
}

void WeightsIndex::load_tensor_data(const std::string& shard_file, uint64_t offset, uint64_t size, void* buffer) const {
    std::string shard_path = model_dir_ + shard_file;
    std::ifstream in(shard_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open shard file for reading: " + shard_path);
    }

    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in) {
        throw std::runtime_error("failed to seek in shard file: " + shard_path);
    }

    in.read(reinterpret_cast<char*>(buffer), static_cast<std::streamsize>(size));
    if (!in) {
        throw std::runtime_error("failed to read tensor data from: " + shard_path);
    }
}

}  // namespace minicpmo

