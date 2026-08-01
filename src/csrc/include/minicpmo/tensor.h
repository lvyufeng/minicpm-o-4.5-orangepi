#pragma once

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace minicpmo {

enum class DType {
    Float16,
    BFloat16,
    Float32,
    Int32,
    Int64,
    UInt8,
    Int8,
};

size_t dtype_size(DType dtype);
aclDataType to_acl_dtype(DType dtype);

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t bytes);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    void allocate(size_t bytes);
    void reset();

    void* data() const { return data_; }
    size_t size_bytes() const { return size_bytes_; }

private:
    void* data_{nullptr};
    size_t size_bytes_{0};
};

// How a tensor's bytes are arranged on device. Almost everything is ND
// (plain row-major). Weights destined for the Cube unit can instead be stored
// FractalNz, the Cube's native tiling, which lets aclnnMm consume them without
// reformatting -- see the note on set_format below.
enum class Format {
    ND,
    FractalNz,
};

class Tensor {
public:
    Tensor() = default;
    Tensor(std::vector<int64_t> shape, DType dtype);

    void allocate();
    void copy_from_host(const void* src, size_t bytes);
    void copy_to_host(void* dst, size_t bytes) const;

    const std::vector<int64_t>& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    size_t numel() const;
    size_t size_bytes() const;
    void* data() const { return buffer_.data(); }

    // `shape()` stays the LOGICAL shape; only the byte layout changes.
    //
    // At M < 16, aclnnMm requests a workspace exactly the size of B and
    // reformats the whole ND weight into the Cube's fractal tiling on every
    // call: 101.19 MB of scratch for a 4096x12288 fp16 weight, ~3x the weight's
    // traffic, and it dominates decode. Handing it a FractalNz weight instead
    // drops the workspace to 0.53 MB and the projection from 12.49 to 4.30 ms
    // (8.06 -> 23.42 GB/s, at the 22.1 GB/s memcpy ceiling), bit-identical
    // results. See tests/bench_nz_handrolled.cpp.
    Format format() const { return format_; }
    void set_format(Format f) { format_ = f; }

private:
    std::vector<int64_t> shape_;
    DType dtype_{DType::Float16};
    Format format_{Format::ND};
    DeviceBuffer buffer_;
};

// FRACTAL_NZ storage dims for a logical [K, N] fp16 matrix: [N/16, K/16, 16, 16].
// Element (k, n) lives at [n/16][k/16][k%16][n%16]. Both dims must be
// multiples of 16.
std::vector<int64_t> fractal_nz_storage_dims(int64_t K, int64_t N);
bool fractal_nz_compatible(int64_t K, int64_t N);

// Permutes a host-side row-major [K, N] fp16 buffer into FRACTAL_NZ order.
void pack_fractal_nz(const uint16_t* src_kn, uint16_t* dst_nz, int64_t K, int64_t N);

}  // namespace minicpmo
