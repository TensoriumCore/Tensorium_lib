#pragma once

#include "CudaRuntime.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace tensorium::cuda {

template <typename T> class DeviceBuffer {
  public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) { resize(count); }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&other) noexcept { swap(other); }

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
        if (this != &other) {
            reset();
            swap(other);
        }
        return *this;
    }

    ~DeviceBuffer() {
        try {
            reset();
        } catch (...) {
        }
    }

    void resize(std::size_t count) {
        if (count == size_)
            return;
        DeviceBuffer tmp;
        if (count > 0) {
            tmp.data_ = static_cast<T *>(tensorium::cuda::malloc_bytes(count * sizeof(T)));
            tmp.size_ = count;
        }
        swap(tmp);
    }

    void reset() {
        if (data_)
            tensorium::cuda::free_bytes(data_);
        data_ = nullptr;
        size_ = 0;
    }

    void swap(DeviceBuffer &other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    [[nodiscard]] std::size_t bytes() const noexcept { return size_ * sizeof(T); }

    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] T *data() noexcept { return data_; }

    [[nodiscard]] const T *data() const noexcept { return data_; }

    void zero() {
        if (!empty())
            tensorium::cuda::memset(data_, 0, bytes());
    }

    void copy_from_host(const T *src, std::size_t count) {
        if (count != size_)
            resize(count);
        if (count == 0)
            return;
        if (!src)
            throw std::invalid_argument("DeviceBuffer::copy_from_host received null source");
        tensorium::cuda::memcpy(data_, src, count * sizeof(T), MemcpyKind::HostToDevice);
    }

    void copy_to_host(T *dst, std::size_t count) const {
        if (count != size_) {
            throw std::invalid_argument("DeviceBuffer::copy_to_host size mismatch");
        }
        if (count == 0)
            return;
        if (!dst)
            throw std::invalid_argument("DeviceBuffer::copy_to_host received null destination");
        tensorium::cuda::memcpy(dst, data_, count * sizeof(T), MemcpyKind::DeviceToHost);
    }

  private:
    T          *data_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace tensorium::cuda
