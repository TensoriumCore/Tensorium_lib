#pragma once

#include "../Utils/MathUtils/MathsUtils.hpp"
#include "../Backend/SIMD/Allocator.hpp"
#include "../Backend/SIMD/CPU_id.hpp"
#include "../Backend/SIMD/SIMD.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>

namespace tensorium {
/**
 * @brief Aligned, SIMD-optimized mathematical vector class for scientific computing.
 *
 * This class implements a 1D container with SIMD-accelerated arithmetic operations
 * including addition, subtraction, scalar multiplication, dot product, and various norms.
 * It uses an aligned memory allocator and supports fused-multiply-add instructions
 * via the architecture-dependent `SimdTraits` specialization.
 *
 * This class is intended to be used in high-performance numerical computing,
 * physics simulations, and linear algebra backends.
 *
 * @tparam K Scalar type (typically float or double).
 */
template <typename K> class Vector {
  public:
    /// Underlying aligned data storage (SIMD-friendly).
    aligned_vector<K> data;
    /** @name Constructors */
    ///@{

    /**
     * @brief Construct from a standard vector.
     * @param vec The std::vector used to initialize the data.
     */
    Vector(const std::vector<K> &vec) : data(vec.begin(), vec.end()) {}
    ///@}

    /** @name Element Access */
    ///@{

    K       &operator[](size_t i) { return data[i]; }
    const K &operator[](size_t i) const { return data[i]; }
    const K &operator()(size_t i) const { return data[i]; }
    K       &operator()(size_t i) { return data[i]; }
    /**
     * @brief Construct an empty vector of size `n`.
     * @param n Number of elements.
     */
    Vector(size_t n) : data(n, K()) {}
    /**
     * @brief Construct from an initializer list.
     * @param init Initializer list.
     */
    Vector(std::initializer_list<K> init) : data(init) {}
    /**
     * @brief Construct a constant vector.
     * @param n Number of elements.
     * @param value Constant value to fill.
     */
    Vector(size_t n, K value) : data(n, value) {}
    ///@}

    /** @name Iterators */
    ///@{
    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }
    ///@}

    /** @name Size and Resizing */
    ///@{

    size_t size() const { return data.size(); }
    void   resize(size_t n) { data.resize(n); }
    ///@}

    /** @name Debug and Utilities */
    ///@{

    /**
     * @brief Print the vector to stdout.
     */

    void print() const {
        std::cout << "Vector size: " << size() << "\n";
        for (K value : data)
            std::cout << "[" << value << "]\n";
    }
    /** @name Basic Operations */
    ///@{

    static Vector<K> canonical(int index, K dx, K dy, K dz) {
        Vector<K> out(3, K(0));
        if (index == 0)
            out(0) = dx;
        else if (index == 1)
            out(1) = dy;
        else if (index == 2)
            out(2) = dz;
        else
            throw std::invalid_argument("Index must be 0, 1, or 2");
        return out;
    }
    /**
     * @brief Subtract two vectors.
     * @param other The vector to subtract.
     * @return A new Vector containing the difference.
     */
    __attribute__((always_inline, hot, flatten)) Vector<K> operator-(const Vector<K> &other) const {
        Vector<K> result(data.size());
        size_t    m = std::min(data.size(), other.data.size());
        for (size_t i = 0; i < m; ++i)
            result[i] = data[i] - other[i];
        for (size_t i = m; i < data.size(); ++i)
            result[i] = data[i];
        return result;
    }

    /**
     * @brief Add another vector to this one (in-place).
     * @param v The vector to add.
     */
    __attribute__((always_inline, hot, flatten)) inline void add(const Vector &v) {
        if (v.size() != size())
            throw std::invalid_argument("Vector sizes do not match");
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;
        const size_t simd_width = Simd::width;
        size_t       n = size();
        size_t       i = 0;

        _mm_prefetch((const char *)&v.data[0], _MM_HINT_T0);

        for (; i + 15 < n; i += 16) {
            reg a0 = Simd::load(&data[i]);
            reg b0 = Simd::load(&v.data[i]);
            a0 = Simd::add(a0, b0);
            Simd::store(&data[i], a0);

            reg a1 = Simd::load(&data[i + simd_width]);
            reg b1 = Simd::load(&v.data[i + simd_width]);
            a1 = Simd::add(a1, b1);
            Simd::store(&data[i + simd_width], a1);
        }

        for (; i < n; ++i)
            data[i] += v.data[i];
    }
    /**
     * @brief Subtract another vector from this one (in-place).
     * @param v The vector to subtract.
     */
    __attribute__((always_inline, hot, flatten)) inline void sub(const Vector &v) {
        if (v.size() != size())
            throw std::invalid_argument("Vector sizes do not match");
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;
        const size_t simd_width = Simd::width;
        size_t       n = size();
        size_t       i = 0;

        _mm_prefetch((const char *)&v.data[0], _MM_HINT_T0);
        for (; i + 15 < n; i += 16) {
            reg a0 = Simd::load(&data[i]);
            reg b0 = Simd::load(&v.data[i]);
            a0 = Simd::sub(a0, b0);
            Simd::store(&data[i], a0);

            reg a1 = Simd::load(&data[i + simd_width]);
            reg b1 = Simd::load(&v.data[i + simd_width]);
            a1 = Simd::sub(a1, b1);
            Simd::store(&data[i + simd_width], a1);
        }

        for (; i < n; ++i)
            data[i] -= v.data[i];
    }
    /**
     * @brief Scale this vector by a scalar (in-place).
     * @param a The scalar value.
     */
    __attribute__((always_inline, hot, flatten)) inline void scl(K a) {
        size_t n = size();
        size_t i = 0;
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;
        const size_t simd_width = Simd::width;
        _mm_prefetch((const char *)&data[0], _MM_HINT_T0);
        reg scalar = Simd::set1(a);
        K *__restrict out = &data[0];

        for (; i + 15 < n; i += 16) {
            reg v0 = Simd::load(out + i);
            v0 = Simd::mul(v0, scalar);
            Simd::store(out + i, v0);

            reg v1 = Simd::load(out + i + simd_width);
            v1 = Simd::mul(v1, scalar);
            Simd::store(out + i + simd_width, v1);
        }

        for (; i < n; ++i)
            out[i] *= a;
    }

    /**
     * @brief Compute the linear combination of vectors with coefficients.
     *
     * @param u List of vectors.
     * @param coefs List of coefficients.
     * @return A vector representing the linear combination: sum(c_i * u_i).
     */
    __attribute__((always_inline, hot, flatten)) static inline Vector<K>
    linear_combination(const std::vector<Vector<K>> &u, const std::vector<K> &coefs) {
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;
        const size_t simd_width = Simd::width;

        if (u.size() != coefs.size())
            throw std::invalid_argument("Mismatched number of vectors and coefficients");
        if (u.empty())
            return Vector<K>(0);

        const size_t n = u[0].size();
        for (const auto &v : u)
            if (v.size() != n)
                throw std::invalid_argument("Vector sizes do not match");

        Vector<K> result(n);

        size_t           i = 0;
        constexpr size_t W = simd_width;
        for (; i + W - 1 < n; i += W) {
            reg acc = Simd::zero();
            for (size_t j = 0; j < u.size(); ++j) {
                reg v = Simd::load(&u[j].data[i]);
                reg c = Simd::set1(coefs[j]);
                acc = Simd::fmadd(v, c, acc);
            }
            Simd::store(&result.data[i], acc);
        }

        for (; i < n; ++i) {
            K acc = K(0);
            for (size_t j = 0; j < u.size(); ++j)
                acc += coefs[j] * u[j].data[i];
            result.data[i] = acc;
        }

        return result;
    }
    /**
     * @brief Linearly interpolate between two vectors.
     *
     * @param a First vector.
     * @param b Second vector.
     * @param t Interpolation factor (0 = a, 1 = b).
     * @return Interpolated vector.
     */
    __attribute__((always_inline, hot, flatten)) static inline Vector<K>
    lerp(const Vector<K> &a, const Vector<K> &b, K t) {
        if (a.size() != b.size())
            throw std::invalid_argument("Vector sizes do not match");
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;
        const size_t simd_width = Simd::width;

        const size_t  n = a.size();
        Vector<K> result(n);

        const reg vt = Simd::set1(t);
        const reg vt1 = Simd::sub(Simd::set1(K(1)), vt);

        size_t i = 0;
        _mm_prefetch((const char *)&a.data[0], _MM_HINT_T0);
        for (; i + 7 < n; i += simd_width) {
            reg va = Simd::load(&a.data[i]);
            reg vb = Simd::load(&b.data[i]);

            reg r = Simd::fmadd(vb, vt, Simd::mul(va, vt1));

            Simd::store_stream(&result.data[i], r);
        }

        for (; i < n; ++i)
            result.data[i] = (K(1) - t) * a.data[i] + t * b.data[i];

        return result;
    }
    ///@}

    /** @name Advanced Operations */
    ///@{

    /**
     * @brief Compute the dot product with another vector.
     * @param v The other vector.
     * @return The scalar dot product.
     */
    __attribute__((always_inline, hot, flatten)) inline K dot(const Vector<K> &v) const {
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;
        const size_t simd_width = Simd::width;
        if (v.size() != size())
            throw std::invalid_argument("Vector sizes do not match");
        const size_t n = size();
        size_t       i = 0;
        reg          acc = Simd::zero();

        const K *__restrict a_ptr = &data[0];
        const K *__restrict b_ptr = &v.data[0];
        _mm_prefetch((const char *)&v.data[0], _MM_HINT_T0);
        for (; i + 7 < n; i += simd_width) {
            reg a = Simd::load(a_ptr + i);
            reg b = Simd::load(b_ptr + i);
            acc = Simd::fmadd(a, b, acc);
        }

        K result = detail::reduce_sum(acc);

        for (; i < n; ++i)
            result += a_ptr[i] * b_ptr[i];

        return result;
    }

    /**
     * @brief Compute the 1-norm (sum of absolute values).
     * @return The L1 norm.
     */
    __attribute__((always_inline, hot, flatten)) inline K norm_1() const {
        size_t n = size();
        size_t i = 0;
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;
        const size_t simd_width = Simd::width;
        reg          acc = Simd::zero();
        reg          sign_mask = Simd::set1(K(-0.0));
        const K *__restrict v_ptr = &data[0];
        _mm_prefetch((const char *)&data[0], _MM_HINT_T0);
        for (; i + 7 < n; i += simd_width) {
            reg v = Simd::load(v_ptr + i);
            reg abs_v = Simd::andnot(sign_mask, v);
            acc = Simd::add(acc, abs_v);
        }

        K result = detail::reduce_sum(acc);

        for (; i < n; ++i)
            result += std::abs(v_ptr[i]);

        return result;
    }
    /**
     * @brief Compute the 2-norm (Euclidean norm).
     * @return The L2 norm.
     */
    __attribute__((always_inline, hot, flatten)) inline K norm_2() const {
        size_t n = size();
        size_t i = 0;
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;
        const size_t simd_width = Simd::width;
        reg          acc = Simd::zero();
        const K *__restrict v_ptr = &data[0];
        _mm_prefetch((const char *)&data[0], _MM_HINT_T0);
        for (; i + 7 < n; i += simd_width) {
            reg v = Simd::load(v_ptr + i);
            acc = Simd::fmadd(v, v, acc);
        }

        K result = detail::reduce_sum(acc);

        for (; i < n; ++i)
            result += v_ptr[i] * v_ptr[i];

        return std::sqrt(result);
    }
    /**
     * @brief Compute the infinity norm (maximum absolute value).
     * @return The L∞ norm.
     */
    __attribute__((always_inline, hot, flatten)) inline K norm_inf() const {
        size_t n = size();
        size_t i = 0;
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;
        const size_t simd_width = Simd::width;
        reg          max_v = Simd::zero();
        reg          sign_mask = Simd::set1(K(-0.0));
        const K *__restrict v_ptr = &data[0];
        _mm_prefetch((const char *)&data[0], _MM_HINT_T0);
        for (; i + 7 < n; i += simd_width) {
            reg v = Simd::load(v_ptr + i);
            reg abs_v = Simd::andnot(sign_mask, v);
            max_v = Simd::max(max_v, abs_v);
        }

        K result = detail::reduce_sum(max_v);

        for (; i < n; ++i)
            result = MathsUtils::_max(result, std::abs(v_ptr[i]));

        return result;
    }
    /**
     * @brief Compute the cosine of the angle between two vectors.
     * @param u First vector.
     * @param v Second vector.
     * @return Cosine of the angle between u and v.
     */

    __attribute__((always_inline, hot, flatten)) static inline K angle_cos(const Vector<K> &u,
                                                                          const Vector<K> &v) {
        if (u.size() != v.size())
            throw std::invalid_argument("Vector sizes do not match");

        const K dot = u.dot(v);
        const K norm_u = u.norm_2();
        const K norm_v = v.norm_2();

        return dot / (norm_u * norm_v);
    }

    /**
     * @brief Compute the cross product between two 3D vectors.
     * @param u First 3D vector.
     * @param v Second 3D vector.
     * @return Resulting 3D vector.
     */

    __attribute__((always_inline, hot, flatten)) static inline Vector<K>
    cross_product(const Vector<K> &u, const Vector<K> &v) {
        if (u.size() != 3 || v.size() != 3)
            throw std::invalid_argument("Cross product is only defined for 3D vectors.");

        Vector<K> r(3);
        if constexpr (std::is_floating_point_v<K>) {
            r.data[0] = std::fma(u.data[1], v.data[2], -u.data[2] * v.data[1]);
            r.data[1] = std::fma(u.data[2], v.data[0], -u.data[0] * v.data[2]);
            r.data[2] = std::fma(u.data[0], v.data[1], -u.data[1] * v.data[0]);
        } else {
            r.data[0] = u.data[1] * v.data[2] - u.data[2] * v.data[1];
            r.data[1] = u.data[2] * v.data[0] - u.data[0] * v.data[2];
            r.data[2] = u.data[0] * v.data[1] - u.data[1] * v.data[0];
        }
        return r;
    }

    Vector<K> &operator+=(const Vector<K> &m) {
        this->add(m);
        return *this;
    }
    Vector<K> &operator-=(const Vector<K> &m) {
        this->sub(m);
        return *this;
    }
    Vector<K> &operator*=(K alpha) {
        this->scl(alpha);
        return *this;
    }
};

template <typename K> inline Vector<K> operator+(const Vector<K> &a, const Vector<K> &b) {
    size_t    n = a.size();
    Vector<K> result(n);
    size_t    m = std::min(n, b.size());
    for (size_t i = 0; i < m; ++i)
        result[i] = a[i] + b[i];
    for (size_t i = m; i < n; ++i)
        result[i] = a[i];
    return result;
}

template <typename K> inline Vector<K> operator-(const Vector<K> &a, const Vector<K> &b) {
    assert(a.size() == b.size());
    Vector<K> result(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] - b[i];
    return result;
}

} // namespace tensorium
