#pragma once
#include "../Core/Vector.hpp"
#include "../Core/Matrix.hpp"
#include "../Core/Tensor.hpp"
#include "../Core/LinearSolver.hpp"
#include "../Core/Derivate.hpp"

namespace morpheus {

	// === VECTOR OPS ===
	template <typename T>
	Vector<T> add_vec(const Vector<T>& a, const Vector<T>& b) {
		Vector<T> result = a;
		result.add(b);
		return result;
	}

	template <typename T>
	Vector<T> sub_vec(const Vector<T>& a, const Vector<T>& b) {
		Vector<T> result = a;
		result.sub(b);
		return result;
	}

	template <typename T>
	Vector<T> scl_vec(const Vector<T>& a, T scalar) {
		Vector<T> result = a;
		result.scl(scalar);
		return result;
	}

	template <typename T> T norm1_vec(const Vector<T>& a)   { return a.norm_1(); }
	template <typename T> T norm2_vec(const Vector<T>& a)   { return a.norm_2(); }
	template <typename T> T normInf_vec(const Vector<T>& a) { return a.norm_inf(); }
	template <typename T> T dot_vec(const Vector<T>& a, const Vector<T>& b) { return a.dot(b); }
	template <typename T> T cosine_vec(const Vector<T>& a, const Vector<T>& b) { return Vector<T>::angle_cos(a, b); }

	template <typename T>
	Vector<T> lerp_vec(const Vector<T>& a, const Vector<T>& b, T t) {
		return Vector<T>::lerp(a, b, t);
	}

	template <typename T>
	Vector<T> linear_combination_vec(const std::vector<Vector<T>>& u, const std::vector<T>& coef) {
		return Vector<T>::linear_combination(u, coef);
	}

	template <typename T>
	Vector<T> cross_vec(const Vector<T>& a, const Vector<T>& b) {
		return Vector<T>::cross_product(a, b);
	}


	// === MATRIX OPS ===
	template <typename T>
	Matrix<T> add_mat(const Matrix<T>& A, const Matrix<T>& B) {
		Matrix<T> result = A;
		result.add(B);
		return result;
	}

	template <typename T>
	Matrix<T> sub_mat(const Matrix<T>& A, const Matrix<T>& B) {
		Matrix<T> result = A;
		result.sub(B);
		return result;
	}

	template <typename T>
	Matrix<T> scl_mat(const Matrix<T>& A, T scalar) {
		Matrix<T> result = A;
		result.scl(scalar);
		return result;
	}

	template <typename T>
	Matrix<T> mul_mat(const Matrix<T>& A, const Matrix<T>& B) {
		return A.mul_mat(B); 
	}

	template <typename T>
		Matrix<T> transpose_mat(const Matrix<T>& A) {
			return A.transpose();
	}

	template <typename T>
		Matrix<T> trace_mat(const Matrix<T>& A) {
			return A.trace();
	}

	template <typename T>
		Vector<T> mul_vec(const Matrix<T>& A, const Vector<T>& x) {
			return A.mul_vec(x);
	}

	// === TENSOR OPS ===

	template<typename K, std::size_t Rank>
		template <size_t I, size_t J>
		Tensor<K, Rank - 2> Tensor<K, Rank>::contract_tensor() const {
			static_assert(I < Rank && J < Rank && I != J, "Invalid contraction indices");
			return contract_simd(*this, I, J);
		}

	template <typename K, std::size_t Rank>
		Tensor<K, Rank> transpose_tensor(const Tensor<K, Rank>& T) {
			return T.transpose_simd();
		}

	// === LINEAR SOLVERS ===
	template <typename T>
		Vector<T> gauss_solve(const Matrix<T>& A, const Vector<T>& b) {
			return solver::Gauss<T>::solve(A, b);
		}

	template <typename T>
		Vector<T> jacobi_solve(const Matrix<T>& A, const Vector<T>& b, T tol = 1e-6, int max_iter = 1000) {
			return solver::Jacobi<T>::solve(A, b, tol, max_iter);
		}
	


	// === DERIVATIVES ===

	template<typename K>
		inline void centered_derivative(const Derivate<K>& input, Derivate<K>& output, size_t axis, K dx) {
			input.centered_derivative(input, output, axis, dx);
		}

	template<typename K, size_t Rank>
		inline void centered_derivative(const DerivateND<K, Rank>& input, DerivateND<K, Rank>& output, size_t axis, K dx) {
			input.centered_derivative(input, output, axis, dx);
		}
}
