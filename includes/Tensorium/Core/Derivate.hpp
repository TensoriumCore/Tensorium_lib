#pragma once
#include "../Utils/MathUtils/MathsUtils.hpp"
#include "../Backend/SIMD/Allocator.hpp"
#include "../Backend/SIMD/CPU_id.hpp"
#include "../Backend/SIMD/SIMD.hpp"
#include "Matrix.hpp"
#include "Tensor.hpp"
#include "Vector.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

/* ************************************************************************** */
/** @file Derivate.hpp
 *  @brief Numerical differentiation operators using SIMD.
 *  @author Louis Touzalin
 *  @ingroup Derivation
 */
/* ************************************************************************** */

namespace tensorium {
/**
 * @brief A 2D aligned matrix for numerical derivatives.
 *
 * @tparam K Scalar type (e.g., float, double).
 */
template <typename K> class Derivate {
  public:
    size_t            rows, cols;
    aligned_vector<K> data;
    size_t            block_size;
    /**
     * @brief Constructor with explicit dimensions.
     *
     * @param r Number of rows.
     * @param c Number of columns.
     */
    Derivate(size_t r, size_t c)
        : rows(r),
          cols(c),
          data(r * c, K()),
          block_size(detect_optimal_block_size()) {
        std::cout << "Auto-selected BLOCK_SIZE = " << block_size << std::endl;
    }
    /**
     * @brief Construct from an existing matrix.
     *
     * @param m Input matrix.
     */
    Derivate(const Matrix<K> &m)
        : rows(m.rows),
          cols(m.cols),
          data(m.data),
          block_size(detect_optimal_block_size()) {}
    /**
     * @brief Mutable access to an element.
     *
     * @param i Row index.
     * @param j Column index.
     * @return Reference to element (i,j).
     */
    K &operator()(size_t i, size_t j) {
        if (i >= rows || j >= cols) {
            std::cerr << "[OOB] operator(): "
                      << "i=" << i << " rows=" << rows << " | j=" << j << " cols=" << cols << "\n";
        }
        assert(i < rows && j < cols);
        return data[i * cols + j];
    }
    /**
     * @brief Const access to an element.
     *
     * @param i Row index.
     * @param j Column index.
     * @return Const reference to element (i,j).
     */
    const K &operator()(size_t i, size_t j) const {
        assert(i < rows && j < cols && "Derivate::operator() const: indice hors bornes");
        return data[i * cols + j];
    }
    /**
     * @brief Total number of elements in the matrix.
     *
     * @return Total size.
     */
    size_t size() const { return rows * cols; }
    /**
     * @brief Compute second-order centered derivative.
     *
     * @param input Input matrix.
     * @param output Output matrix.
     * @param axis Differentiation axis (0 = rows, 1 = cols).
     * @param dx Grid spacing.
     */
    __attribute__((always_inline, hot, flatten)) inline void
    centered_derivative(const Derivate<K> &input, Derivate<K> &output, size_t axis, K dx) const {
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;
        const size_t simd_width = Simd::width;
        const K      inv_2dx = K(1) / (K(2) * dx);

        if (axis == 1) {
#pragma omp parallel for
            for (size_t i = 0; i < input.rows; ++i) {
                output(i, 0) = K(0);

                size_t       j = 1;
                const size_t simd_end =
                    (input.cols > simd_width + 1) ? (input.cols - simd_width - 1) : 1;
                for (; j < simd_end; j += simd_width) {
                    if (j + simd_width > input.cols - 1)
                        break;

                    const K *left_ptr = &input(i, j - 1);
                    const K *right_ptr = &input(i, j + 1);
                    K       *out_ptr = &output(i, j);

                    reg left = Simd::loadu(left_ptr);
                    reg right = Simd::loadu(right_ptr);
                    reg diff = Simd::sub(right, left);
                    reg res = Simd::mul(diff, Simd::set1(inv_2dx));
                    Simd::storeu(out_ptr, res);
                }

                for (; j < input.cols - 1; ++j)
                    output(i, j) = (input(i, j + 1) - input(i, j - 1)) * inv_2dx;

                output(i, input.cols - 1) = K(0);
            }
        }

        else if (axis == 0) {
#pragma omp parallel for
            for (size_t i = 1; i < input.rows - 1; ++i)
                for (size_t j = 0; j < input.cols; ++j)
                    output(i, j) = (input(i + 1, j) - input(i - 1, j)) * inv_2dx;

            for (size_t j = 0; j < input.cols; ++j) {
                output(0, j) = K(0);
                output(input.rows - 1, j) = K(0);
            }
        }

        else {
            std::cerr << "[centered_derivative] Invalid axis: must be 0 or 1\n";
        }
    }
    /**
     * @brief Compute fourth-order centered derivative.
     *
     * @param input Input matrix.
     * @param output Output matrix.
     * @param axis Differentiation axis.
     * @param dx Grid spacing.
     */
    __attribute__((always_inline, hot, flatten)) inline void
    centered_derivative_order4(const Derivate<K> &input, Derivate<K> &output, size_t axis, K dx) {
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;
        constexpr size_t W = Simd::width;

        const K   inv_12dx = K(1) / (K(12) * dx);
        const reg inv = Simd::set1(inv_12dx);
        const reg eight = Simd::set1(8.0f);

        if (axis == 1) {
#pragma omp parallel for
            for (size_t i = 0; i < input.rows; ++i) {
                output(i, 0) = K(0);
                output(i, 1) = K(0);

                size_t j = 2;
                for (; j + W <= input.cols - 2; j += W) {
                    const K *ptr_m2 = &input(i, j - 2);
                    const K *ptr_m1 = &input(i, j - 1);
                    const K *ptr_p1 = &input(i, j + 1);
                    const K *ptr_p2 = &input(i, j + 2);

                    reg fm2 = Simd::loadu(ptr_m2);
                    reg fm1 = Simd::loadu(ptr_m1);
                    reg fp1 = Simd::loadu(ptr_p1);
                    reg fp2 = Simd::loadu(ptr_p2);

                    reg term1 = Simd::mul(fp1, eight);
                    reg term2 = Simd::mul(fp2, Simd::set1(1.0f));
                    reg term3 = Simd::mul(fm1, eight);

                    reg num = Simd::sub(Simd::add(fm2, term1), term2);
                    num = Simd::sub(num, term3);
                    reg res = Simd::mul(num, inv);

                    Simd::storeu(&output(i, j), res);
                }

                for (; j < input.cols - 2; ++j) {
                    output(i, j) = (input(i, j - 2) - 8 * input(i, j - 1) + 8 * input(i, j + 1) -
                                    input(i, j + 2)) *
                                   inv_12dx;
                }

                output(i, input.cols - 2) = K(0);
                output(i, input.cols - 1) = K(0);
            }
        } else if (axis == 0) {
#pragma omp parallel for
            for (size_t j = 0; j < input.cols; ++j) {
                output(0, j) = K(0);
                output(1, j) = K(0);

                size_t i = 2;
                for (; i + W <= input.rows - 2; i += W) {
                    const K *ptr_m2 = &input(i - 2, j);
                    const K *ptr_m1 = &input(i - 1, j);
                    const K *ptr_p1 = &input(i + 1, j);
                    const K *ptr_p2 = &input(i + 2, j);

                    reg fm2 = Simd::loadu(ptr_m2);
                    reg fm1 = Simd::loadu(ptr_m1);
                    reg fp1 = Simd::loadu(ptr_p1);
                    reg fp2 = Simd::loadu(ptr_p2);

                    reg term1 = Simd::mul(fp1, eight);
                    reg term2 = Simd::mul(fp2, Simd::set1(1.0f));
                    reg term3 = Simd::mul(fm1, eight);

                    reg num = Simd::sub(Simd::add(fm2, term1), term2);
                    num = Simd::sub(num, term3);
                    reg res = Simd::mul(num, inv);

                    Simd::storeu(&output(i, j), res);
                }

                for (; i < input.rows - 2; ++i) {
                    output(i, j) = (input(i - 2, j) - 8 * input(i - 1, j) + 8 * input(i + 1, j) -
                                    input(i + 2, j)) *
                                   inv_12dx;
                }

                output(input.rows - 2, j) = K(0);
                output(input.rows - 1, j) = K(0);
            }
        } else {
            std::cerr << "[SIMD Order4] Invalid axis: must be 0 or 1\n";
        }
    }
};

/**
 * @brief A multi-dimensional aligned tensor for numerical derivatives.
 *
 * @tparam K Scalar type.
 * @tparam Rank Tensor rank (dimensionality).
 */
template <typename K, size_t Rank> class DerivateND {
  public:
    std::array<size_t, Rank> shape;
    aligned_vector<K>        data;
    size_t                   block_size;
    /**
     * @brief Construct a tensor with a given shape.
     *
     * @param dims Tensor dimensions.
     */
    DerivateND(const std::array<size_t, Rank> &dims)
        : shape(dims),
          data(std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>()),
               K()),
          block_size(detect_optimal_block_size()) {
        std::cout << "Auto-selected BLOCK_SIZE = " << block_size << std::endl;
    }

    inline size_t flatten_index(const std::array<size_t, Rank> &indices) const {
        size_t index = 0, stride = 1;
        for (int i = Rank - 1; i >= 0; --i) {
            index += indices[i] * stride;
            stride *= shape[i];
        }
        return index;
    }
    /**
     * @brief Mutable access to a tensor element.
     *
     * @param indices Multi-index.
     * @return Reference to the element.
     */
    inline K &operator()(const std::array<size_t, Rank> &indices) {
        return data[flatten_index(indices)];
    }
    /**
     * @brief Const access to a tensor element.
     *
     * @param indices Multi-index.
     * @return Const reference to the element.
     */
    inline const K &operator()(const std::array<size_t, Rank> &indices) const {
        return data[flatten_index(indices)];
    }
    /**
     * @brief Total number of elements in the tensor.
     *
     * @return Total size.
     */
    inline size_t size() const { return data.size(); }
    /**
     * @brief Compute second-order centered derivative along an axis.
     *
     * @param input Input tensor.
     * @param output Output tensor.
     * @param axis Axis to differentiate.
     * @param dx Grid spacing.
     */
    __attribute__((always_inline, hot, flatten)) inline void
    centered_derivative(const DerivateND<K, Rank> &input, DerivateND<K, Rank> &output, size_t axis,
                        K dx) const {
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;
        const size_t simd_width = Simd::width;

        const auto  &shape = input.shape;
        const size_t total = input.size();
        const K      inv_2dx = K(1) / (K(2) * dx);
        const reg    inv2dx = Simd::set1(inv_2dx);

        std::array<size_t, Rank> strides;
        strides[Rank - 1] = 1;
        for (int i = Rank - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape[i + 1];

        const size_t stride_axis = strides[axis];
        const size_t dim_axis = shape[axis];

#pragma omp parallel for schedule(static)
        for (size_t flat = 0; flat < total; flat += simd_width) {
            bool safe = true;

            for (size_t offset = 0; offset < simd_width; ++offset) {
                if (flat + offset >= total) {
                    safe = false;
                    break;
                }

                const size_t coord_axis = ((flat + offset) / stride_axis) % dim_axis;
                if (coord_axis == 0 || coord_axis >= dim_axis - 1) {
                    safe = false;
                    break;
                }
            }

            if (!safe) {
                for (size_t offset = 0; offset < simd_width && flat + offset < total; ++offset) {
                    const size_t f = flat + offset;
                    const size_t coord_axis = (f / stride_axis) % dim_axis;

                    if (coord_axis == 0 || coord_axis >= dim_axis - 1) {
                        output.data[f] = K(0);
                        continue;
                    }
                    output.data[f] =
                        (input.data[f + stride_axis] - input.data[f - stride_axis]) * inv_2dx;
                }
                continue;
            }

            const K *ptr_fwd = input.data.data() + flat + stride_axis;
            const K *ptr_back = input.data.data() + flat - stride_axis;
            K       *out_ptr = output.data.data() + flat;

            reg forward = Simd::loadu(ptr_fwd);
            reg backward = Simd::loadu(ptr_back);
            reg diff = Simd::sub(forward, backward);
            reg result = Simd::mul(diff, inv2dx);
            Simd::storeu(out_ptr, result);
        }
    }
    /**
     * @brief Compute fourth-order centered derivative along an axis.
     *
     * @param input Input tensor.
     * @param output Output tensor.
     * @param axis Axis to differentiate.
     * @param dx Grid spacing.
     */
    __attribute__((always_inline, hot, flatten)) inline void
    centered_derivative_order4_rank(const DerivateND<K, Rank> &input, DerivateND<K, Rank> &output,
                                    size_t axis, K dx) const {
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;
        constexpr size_t W = Simd::width;

        const auto  &shape = input.shape;
        const size_t total = input.size();
        const K      inv_12dx = K(1) / (K(12) * dx);
        const reg    inv = Simd::set1(inv_12dx);

        std::array<size_t, Rank> strides;
        strides[Rank - 1] = 1;
        for (int i = Rank - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape[i + 1];

        const size_t stride_axis = strides[axis];
        const size_t dim_axis = shape[axis];

#pragma omp parallel for schedule(static)
        for (size_t flat = 0; flat < total; flat += W) {
            bool safe = true;

            for (size_t offset = 0; offset < W; ++offset) {
                if (flat + offset >= total) {
                    safe = false;
                    break;
                }

                const size_t coord_axis = ((flat + offset) / stride_axis) % dim_axis;
                if (coord_axis < 2 || coord_axis >= dim_axis - 2) {
                    safe = false;
                    break;
                }
            }

            if (!safe) {
                for (size_t offset = 0; offset < W && flat + offset < total; ++offset) {
                    size_t f = flat + offset;
                    size_t coord_axis = (f / stride_axis) % dim_axis;
                    if (coord_axis < 2 || coord_axis >= dim_axis - 2) {
                        output.data[f] = K(0);
                        continue;
                    }

                    K fm2 = input.data[f - 2 * stride_axis];
                    K fm1 = input.data[f - stride_axis];
                    K fp1 = input.data[f + stride_axis];
                    K fp2 = input.data[f + 2 * stride_axis];

                    output.data[f] = (-fp2 + 8 * fp1 - 8 * fm1 + fm2) * inv_12dx;
                }
                continue;
            }

            const K *ptr_m2 = input.data.data() + flat - 2 * stride_axis;
            const K *ptr_m1 = input.data.data() + flat - stride_axis;
            const K *ptr_p1 = input.data.data() + flat + stride_axis;
            const K *ptr_p2 = input.data.data() + flat + 2 * stride_axis;
            K       *out_ptr = output.data.data() + flat;

            reg fm2 = Simd::loadu(ptr_m2);
            reg fm1 = Simd::loadu(ptr_m1);
            reg fp1 = Simd::loadu(ptr_p1);
            reg fp2 = Simd::loadu(ptr_p2);

            reg num = Simd::add(
                fm2, Simd::sub(Simd::mul(fp1, Simd::set1(8)), Simd::mul(fp2, Simd::set1(1))));
            num = Simd::sub(num, Simd::mul(fm1, Simd::set1(8)));
            reg res = Simd::mul(num, inv);

            Simd::storeu(out_ptr, res);
        }
    }
};
/**
 * @brief Richardson extrapolation for vectors or containers.
 *
 * @tparam Container Vector-like container type.
 * @param plus_h f(x + h)
 * @param minus_h f(x - h)
 * @param plus_half_h f(x + h/2)
 * @param minus_half_h f(x - h/2)
 * @param h Step size.
 * @return Richardson-extrapolated derivative.
 */
template <typename Container>
inline Container richardson_derivative_container(const Container &plus_h, const Container &minus_h,
                                                 const Container &plus_half_h,
                                                 const Container &minus_half_h, double h) {
    assert(plus_h.size() == minus_h.size());
    assert(plus_h.size() == plus_half_h.size());
    assert(plus_h.size() == minus_half_h.size());

    Container out(plus_h);
#pragma omp parallel for
    for (size_t i = 0; i < plus_h.size(); ++i) {
        auto diff_h = (plus_h[i] - minus_h[i]) / (2.0 * h);
        auto diff_half = (plus_half_h[i] - minus_half_h[i]) / h;
        out[i] = (4.0 * diff_half - diff_h) / 3.0;
    }
    return out;
}
/**
 * @brief Richardson extrapolation for scalar values.
 *
 * @tparam T Scalar type.
 * @param plus_h f(x + h)
 * @param minus_h f(x - h)
 * @param plus_half_h f(x + h/2)
 * @param minus_half_h f(x - h/2)
 * @param h Step size.
 * @return Richardson-extrapolated scalar derivative.
 */
template <typename T>
inline T richardson_derivative(const T &plus_h, const T &minus_h, const T &plus_half_h,
                               const T &minus_half_h, double h) {
    T diff_h = (plus_h - minus_h) / (2.0 * h);
    T diff_half = (plus_half_h - minus_half_h) / h;
    return (4.0 * diff_half - diff_h) / 3.0;
}

} // namespace tensorium
