#pragma once

#include "DeviceBuffer.hpp"

#include <Tensorium/Core/Matrix.hpp>

namespace tensorium::cuda {

template <typename T, bool RowMajor = false> class MatrixCUDA {
  public:
    MatrixCUDA() = default;

    MatrixCUDA(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), buffer_(rows * cols) {}

    explicit MatrixCUDA(const tensorium::Matrix<T, RowMajor> &host) { copy_from_host(host); }

    void resize(std::size_t rows, std::size_t cols) {
        rows_ = rows;
        cols_ = cols;
        buffer_.resize(rows * cols);
    }

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }

    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }

    [[nodiscard]] std::size_t size() const noexcept { return buffer_.size(); }

    [[nodiscard]] std::size_t bytes() const noexcept { return buffer_.bytes(); }

    [[nodiscard]] bool empty() const noexcept { return buffer_.empty(); }

    [[nodiscard]] T *data() noexcept { return buffer_.data(); }

    [[nodiscard]] const T *data() const noexcept { return buffer_.data(); }

    [[nodiscard]] std::size_t leading_dimension() const noexcept { return RowMajor ? cols_ : rows_; }

    void fill_zero() { buffer_.zero(); }

    void copy_from_host(const tensorium::Matrix<T, RowMajor> &host) {
        rows_ = host.rows;
        cols_ = host.cols;
        buffer_.copy_from_host(host.data.data(), host.size());
    }

    void copy_to_host(tensorium::Matrix<T, RowMajor> &host) const {
        if (host.rows != rows_ || host.cols != cols_) {
            host = tensorium::Matrix<T, RowMajor>(rows_, cols_);
        }
        buffer_.copy_to_host(host.data.data(), size());
    }

    [[nodiscard]] tensorium::Matrix<T, RowMajor> to_host() const {
        tensorium::Matrix<T, RowMajor> host(rows_, cols_);
        copy_to_host(host);
        return host;
    }

  private:
    std::size_t     rows_ = 0;
    std::size_t     cols_ = 0;
    DeviceBuffer<T> buffer_;
};

} // namespace tensorium::cuda
