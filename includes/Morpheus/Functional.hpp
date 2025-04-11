
#pragma once
#include "Vector.hpp"
#include "Matrix.hpp"

namespace morpheus {

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

	// === MATRIX ===
	template <typename T>
		Matrix<T> add(const Matrix<T>& A, const Matrix<T>& B) {
			Matrix<T> result = A;
			result.add(B);
			return result;
		}

	template <typename T>
		Matrix<T> sub(const Matrix<T>& A, const Matrix<T>& B) {
			Matrix<T> result = A;
			result.sub(B);
			return result;
		}

	template <typename T>
		Matrix<T> scl(const Matrix<T>& A, T scalar) {
			Matrix<T> result = A;
			result.scl(scalar);
			return result;
		}

} 
