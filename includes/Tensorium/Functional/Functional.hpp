#pragma once
#include "../Backend/CPU_Kernels/MatrixKernel.hpp"
#include "../Core/Derivate.hpp"
#include "../Core/LinearSolver.hpp"
#include "../Core/Matrix.hpp"
#include "../Core/Spectral.hpp"
#include "../Core/Tensor.hpp"
#include "../Core/Vector.hpp"

namespace tensorium {
// === VECTOR OPS ===

/*
 * @brief Add two vectors
 * @param a First vector
 * @param b Second vector
 * @return Resulting vector
 */

template <typename T> Vector<T> add_vec(const Vector<T> &a, const Vector<T> &b) {
    Vector<T> result = a;
    result.add(b);
    return result;
}

/*
 * @brief Subtract two vectors
 * @param a First vector
 * @param b Second vector
 * @return Resulting vector
 */

template <typename T> Vector<T> sub_vec(const Vector<T> &a, const Vector<T> &b) {
    Vector<T> result = a;
    result.sub(b);
    return result;
}

/*
 * @brief Scale a vector by a scalar
 * @param a Vector to scale
 * @param scalar Scalar value
 * @return Resulting vector
 */

template <typename T> Vector<T> scl_vec(const Vector<T> &a, T scalar) {
    Vector<T> result = a;
    result.scl(scalar);
    return result;
}

/*
 * @brief Normalize a vector
 * @param a Vector to normalize
 * @return Normalized vector
 */

template <typename T> T norm1_vec(const Vector<T> &a) { return a.norm_1(); }

/*
 * @brief Normalize 2 a vector
 * @param a Vector to normalize
 * @return Normalized vector
 */

template <typename T> T norm2_vec(const Vector<T> &a) { return a.norm_2(); }

/*
 * @brief Normalize inf a vector
 * @param a Vector to normalize
 * @return Normalized vector
 */

template <typename T> T normInf_vec(const Vector<T> &a) { return a.norm_inf(); }

/*
 * @brief Dot product of two vectors
 * @param a First vector
 * @param b Second vector
 * @return Dot product result
 */

template <typename T> T dot_vec(const Vector<T> &a, const Vector<T> &b) { return a.dot(b); }
/*
 * Cosine of the angle between two vectors
 * @param a First vector
 * @param b Second vector
 * @return Cosine of the angle
 */

template <typename T> T cosine_vec(const Vector<T> &a, const Vector<T> &b) {
    return Vector<T>::angle_cos(a, b);
}
/*
 * Lerp between two vectors
 * @param a First vector
 * @param b Second vector
 * @param t Interpolation factor
 * @return Interpolated vector
 */

template <typename T> Vector<T> lerp_vec(const Vector<T> &a, const Vector<T> &b, T t) {
    return Vector<T>::lerp(a, b, t);
}

/*
 * Linear combination of multiple vectors
 * @param u Vector of vectors
 * @param coef Coefficients for the linear combination
 * @return Resulting vector
 */

template <typename T>
Vector<T> linear_combination_vec(const std::vector<Vector<T>> &u, const std::vector<T> &coef) {
    return Vector<T>::linear_combination(u, coef);
}

/*
 * Cross product of two vectors
 * @param a First vector
 * @param b Second vector
 * @return Cross product result
 */

template <typename T> Vector<T> cross_vec(const Vector<T> &a, const Vector<T> &b) {
    return Vector<T>::cross_product(a, b);
}

// === MATRIX OPS ===

/*
 * @brief Add two matrices
 * @param A First matrix
 * @param B Second matrix
 * @return Resulting matrix
 */

template <typename T> Matrix<T> add_mat(const Matrix<T> &A, const Matrix<T> &B) {
    Matrix<T> result = A;
    result.add(B);
    return result;
}

/*
 * @brief Subtract two matrices
 * @param A First matrix
 * @param B Second matrix
 * @return Resulting matrix
 */

template <typename T> Matrix<T> sub_mat(const Matrix<T> &A, const Matrix<T> &B) {
    Matrix<T> result = A;
    result.sub(B);
    return result;
}

/*
 * @brief Scale a matrix by a scalar
 * @param A Matrix to scale
 * @param scalar Scalar value
 * @return Resulting matrix
 */

template <typename T> Matrix<T> scl_mat(const Matrix<T> &A, T scalar) {
    Matrix<T> result = A;
    result.scl(scalar);
    return result;
}

template <typename T> Matrix<T> lerp_mat(const Matrix<T> &A, const Matrix<T> &B, T t) {
    Matrix<T> result(A.rows, A.cols);
    result.lerp(A, B, t);
    return result;
}

/*
 * @brief Matrix multiplication
 * @param A First matrix
 * @param B Second matrix
 * @return Resulting matrix
 * @note Assumes A.cols() == B.rows()
 * @note Uses SIMD for optimization
 * @note OpenMP parallelization for large matrices, for small matrices, it is not worth the overhead
 * @note (For matrix 4x4, 8x8 and 16x16, there's a SIMD fallback to ensure L1 and L2 cache storage)
 */

template <typename T> Matrix<T> mul_mat(const Matrix<T> &A, const Matrix<T> &B) {
    // if (A.rows == 2 && A.cols == 2 && B.rows == 2 && B.cols == 2) {
    // 	const MatrixKernel<T> kernelA(A);
    // 	const MatrixKernel<T> kernelB(B);
    // 	return kernelA.mul_mat2x2(kernelB);
    // }

    if (A.rows == 3 && A.cols == 3 && B.rows == 3 && B.cols == 3) {
        const MatrixKernel<T> kernelA(A);
        const MatrixKernel<T> kernelB(B);
        return kernelA.mul_mat3x3(kernelB);
    }

    if (A.rows == 4 && A.cols == 4 && B.rows == 4 && B.cols == 4) {
        const MatrixKernel<T> kernelA(A);
        const MatrixKernel<T> kernelB(B);
        return kernelA.mul_mat4x4(kernelB);
    }

    if (A.rows == 8 && A.cols == 8 && B.rows == 8 && B.cols == 8) {
        const MatrixKernel<T> kernelA(A);
        const MatrixKernel<T> kernelB(B);
        return kernelA.mul_mat8x8(kernelB);
    }

    if (A.rows == 16 && A.cols == 16 && B.rows == 16 && B.cols == 16) {
        const MatrixKernel<T> kernelA(A);
        const MatrixKernel<T> kernelB(B);
        return kernelA.mul_mat16x16(kernelB);
    }

    if (A.rows == 32 && A.cols == 32 && B.rows == 32 && B.cols == 32) {
        const MatrixKernel<T> kernelA(A);
        const MatrixKernel<T> kernelB(B);
        return kernelA.mul_mat32x32(kernelB);
    }
    //
    //
    // if (A.rows == 64 && A.cols == 64 && B.rows == 64 && B.cols == 64) {
    // 	const MatrixKernel<T> kernelA(A);
    // 	const MatrixKernel<T> kernelB(B);
    // 	return kernelA.mul_mat64x64(kernelB);
    // }
    //
    //

    return A._mul_mat(B);
}

/*
 * @brief Matrix transpose
 * @param A Matrix to transpose
 */

template <typename T> Matrix<T> transpose_mat(const Matrix<T> &A) { return A.transpose(); }

/*
 * @brief Trace of a matrix
 * @param A Matrix to trace
 */

template <typename T> Matrix<T> trace_mat(const Matrix<T> &A) { return A.trace(); }

/*
 * @brief Multiply a matrix by a vector
 * @param A Matrix
 * @param x Vector
 * @return Resulting vector
 * @note Assumes A.cols() == x.size()
 */

template <typename T> Vector<T> mul_vec(const Matrix<T> &A, const Vector<T> &x) {
    return A.mul_vec(x);
}

/*
 * @brief Inverse of a matrix
 * @param A Matrix to invert
 * @return Inverted matrix
 * @note Assumes A is square
 * @note Uses LU decomposition for inversion
 */

template <typename T> Matrix<T> inverse_mat(const Matrix<T> &A) { return A.inverse(); }

template <typename T> T det_mat(const Matrix<T> &A) { return A.det(); }

template <typename T> size_t rank_mat(const Matrix<T> &A) { return A.rank(); }

// === TENSOR OPS ===
template <typename K, std::size_t Rank>
template <size_t I, size_t J>
Tensor<K, Rank - 2> Tensor<K, Rank>::contract_tensor() const {
    static_assert(I < Rank && J < Rank && I != J, "Invalid contraction indices");
    return contract_simd(*this, I, J);
}

template <typename K, std::size_t Rank> Tensor<K, Rank> transpose_tensor(const Tensor<K, Rank> &T) {
    return T.transpose_simd();
}

template <typename K, size_t R1, size_t R2>
inline Tensor<K, R1 + R2> mul_tensor(const Tensor<K, R1> &A, const Tensor<K, R2> &B) {
    return Tensor<K, R1>::template tensor_product<R1, R2>(A, B);
}

template <size_t I, size_t J, typename K, std::size_t Rank>
Tensor<K, Rank - 2> contract_tensor(const Tensor<K, Rank> &T) {
    static_assert(I < Rank && J < Rank && I != J, "Invalid contraction indices");
    return T.template contract_tensor<I, J>();
}
// === LINEAR SOLVERS ===
template <typename T> Vector<T> gauss_solve(const Matrix<T> &A, const Vector<T> &b) {
    return solver::Gauss<T>::solve(A, b);
}

template <typename T>
Vector<T> jacobi_solve(const Matrix<T> &A, const Vector<T> &b, T tol = 1e-6, int max_iter = 1000) {
    return solver::Jacobi<T>::solve(A, b, tol, max_iter);
}

template <typename T>
inline void row_echelon(Matrix<T> &A, Vector<T> *b = nullptr, T eps = T(1e-12)) {
    solver::Gauss<T>::raw_row_echelon(A, b, eps);
}

template <typename K>
inline void centered_derivative(const Derivate<K> &input, Derivate<K> &output, size_t axis, K dx) {
    input.centered_derivative(input, output, axis, dx);
}

template <typename K>
inline void centered_derivative_order4(const Derivate<K> &input, Derivate<K> &output, size_t axis,
                                       K dx) {
    input.centered_derivative_order4(input, output, axis, dx);
}

template <typename K, size_t Rank>
inline void centered_derivative(const DerivateND<K, Rank> &input, DerivateND<K, Rank> &output,
                                size_t axis, K dx) {
    input.centered_derivative(input, output, axis, dx);
}

template <typename K, size_t Rank>
inline void centered_derivative_order4(const DerivateND<K, Rank> &input,
                                       DerivateND<K, Rank> &output, size_t axis, K dx) {
    input.centered_derivative_order4_rank(input, output, axis, dx);
}

//=== SPECTRAL METHODS ===

template <typename T> inline void forwardFFT(tensorium::Vector<std::complex<T>> &data) {
    SpectralFFT<T>::forward(data);
}

template <typename T> inline void backwardFFT(tensorium::Vector<std::complex<T>> &data) {
    SpectralFFT<T>::backward(data);
}

template <typename T> inline void backwardFFP(tensorium::Vector<std::complex<T>> &data) {
    SpectralFFT<T>::backward(data);
}

} // namespace tensorium
