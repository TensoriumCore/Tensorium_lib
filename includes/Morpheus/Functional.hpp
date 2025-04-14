#pragma once
#include "Vector.hpp"
#include "Matrix.hpp"
#include "Tensor.h"

namespace morpheus {

	// === VECTOR OPS ===
	template <typename T>
	Vector<T> add(const Vector<T>& a, const Vector<T>& b) {
		Vector<T> result = a;
		result.add(b);
		return result;
	}

	template <typename T>
	Vector<T> sub(const Vector<T>& a, const Vector<T>& b) {
		Vector<T> result = a;
		result.sub(b);
		return result;
	}

	template <typename T>
	Vector<T> scl(const Vector<T>& a, T scalar) {
		Vector<T> result = a;
		result.scl(scalar);
		return result;
	}

	template <typename T> T norm1(const Vector<T>& a)   { return a.norm_1(); }
	template <typename T> T norm2(const Vector<T>& a)   { return a.norm_2(); }
	template <typename T> T normInf(const Vector<T>& a) { return a.norm_inf(); }
	template <typename T> T dot(const Vector<T>& a, const Vector<T>& b) { return a.dot(b); }
	template <typename T> T cosine(const Vector<T>& a, const Vector<T>& b) { return Vector<T>::angle_cos(a, b); }

	template <typename T>
	Vector<T> lerp(const Vector<T>& a, const Vector<T>& b, T t) {
		return Vector<T>::lerp(a, b, t);
	}

	template <typename T>
	Vector<T> linear_combination(const std::vector<Vector<T>>& u, const std::vector<T>& coef) {
		return Vector<T>::linear_combination(u, coef);
	}

	template <typename T>
	Vector<T> cross(const Vector<T>& a, const Vector<T>& b) {
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

	// === TENSOR OPS ===
	template<typename K, std::size_t Rank>
		template <size_t I, size_t J>
		Tensor<K, Rank - 2> Tensor<K, Rank>::contract() const {
			static_assert(I < Rank && J < Rank && I != J, "Invalid contraction indices");
			return contract_simd(*this, I, J);
		}


}
