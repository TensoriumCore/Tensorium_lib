#pragma once

#include "DeviceBuffer.hpp"

#include <Tensorium/Core/Vector.hpp>

namespace tensorium::cuda {

template <typename T> class VectorCUDA {
  public:
    VectorCUDA() = default;

    explicit VectorCUDA(std::size_t size) : buffer_(size) {}

    explicit VectorCUDA(const tensorium::Vector<T> &host) { copy_from_host(host); }

    void resize(std::size_t size) { buffer_.resize(size); }

    [[nodiscard]] std::size_t size() const noexcept { return buffer_.size(); }

    [[nodiscard]] std::size_t bytes() const noexcept { return buffer_.bytes(); }

    [[nodiscard]] bool empty() const noexcept { return buffer_.empty(); }

    [[nodiscard]] T *data() noexcept { return buffer_.data(); }

    [[nodiscard]] const T *data() const noexcept { return buffer_.data(); }

    void fill_zero() { buffer_.zero(); }

    void copy_from_host(const tensorium::Vector<T> &host) {
        buffer_.copy_from_host(host.data.data(), host.size());
    }

    void copy_from_host(const T *src, std::size_t count) { buffer_.copy_from_host(src, count); }

    void copy_to_host(tensorium::Vector<T> &host) const {
        if (host.size() != size())
            host.resize(size());
        buffer_.copy_to_host(host.data.data(), size());
    }

    [[nodiscard]] tensorium::Vector<T> to_host() const {
        tensorium::Vector<T> host(size());
        copy_to_host(host);
        return host;
    }

  private:
    DeviceBuffer<T> buffer_;
};

} // namespace tensorium::cuda
