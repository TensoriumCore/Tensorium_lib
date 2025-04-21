#pragma once
#include "Matrix.hpp"
#include "Vector.hpp"
#include "../SIMD/SIMD.hpp" 
#include "../SIMD/CPU_id.hpp" 
#include "../SIMD/Allocator.hpp"

namespace morpheus::solver {
	template<typename K>
		class Jacobi;

	template<typename K>
		class Gauss {
			public:
				size_t rows() const;
				size_t block_size;
				aligned_vector<K> data;
				__attribute__((always_inline, hot, flatten))
					static inline Vector<K> solve(const Matrix<K>& A_in, const Vector<K>& b_in) {

						static_assert(std::is_floating_point<K>::value, "");
						const size_t n = A_in.rows;

						assert(n == A_in.cols && n == b_in.size());
						if (A_in.rows >= 2048 || A_in.cols >= 2048)
							Jacobi<K>::solve(A_in, b_in);

						Matrix<double> M(n, n);
						Vector<double> B(n);
						for(size_t i = 0; i < n; ++i){
							B[i] = b_in[i];
							for(size_t j = 0; j < n; ++j)
								M(i, j) = A_in(i, j);
						}

						Vector<double> x_d(n);
						using SimdD = simd::SimdTraits<double, DefaultISA>;
						using regD   = typename SimdD::reg;
						const size_t W = SimdD::width;

						for(size_t i=0;i<n;++i){
							size_t piv = i;
							double maxv = std::abs(M(i,i));
							for(size_t r=i+1;r<n;++r){
								double v = std::abs(M(r,i));
								if(v > maxv) { maxv=v; piv=r; }
							}
							if(maxv < 1e-12)
								throw std::runtime_error("Pivot trop petit");
							if(piv != i) {
								M.swap_rows(i, piv);
								std::swap(B[i], B[piv]);
							}

							constexpr size_t TILE = UNROLL;  
#pragma omp parallel for schedule(dynamic, TILE)
							for(size_t j = i + 1;j < n; ++j) {
								double *rowj = &M(j, 0), * rowi = &M(i, 0);
								double f = rowj[i] / rowi[i];
								rowj[i] = 0.0;

								regD fv = SimdD::set1(f);
								size_t k= i + 1;
								for(; k + 4 * W <= n; k += 4 * W) {
									for(int t = 0; t < 4; ++t){
										size_t off = k + t * W;
										regD mj = SimdD::loadu(rowj + off);
										regD mi = SimdD::loadu(rowi + off);
										mj = SimdD::sub(mj, SimdD::mul(fv, mi));
										SimdD::storeu(rowj + off, mj);
									}
								}
								for(;k + W <= n; k += W) {
									regD mj = SimdD::loadu(rowj + k);
									regD mi = SimdD::loadu(rowi + k);
									mj = SimdD::sub(mj, SimdD::mul(fv, mi));
									SimdD::storeu(rowj + k, mj);
								}
								for(; k < n; ++k ) {
									rowj[k] -= f * rowi[k];
								}
								B[j] -= f * B[i];
							}
						}

						for(ssize_t i = n - 1; i >= 0; --i) {
							double *rowi = &M(i, 0);
							regD acc = SimdD::setzero();
							size_t j = i + 1;
							for(; j + W <= n; j += W) {
								regD u = SimdD::loadu(rowi + j);
								regD xv= SimdD::loadu(&x_d[j]);
								acc = SimdD::fmadd(u, xv, acc);
							}
							double sum = SimdD::horizontal_add(acc);
							for(; j < n; ++j) sum += rowi[j] * x_d[j];
							x_d[i] = (B[i] - sum) / rowi[i];
						}

						Vector<K> x(n);
						for(size_t i = 0; i < n; ++i) x[i] = static_cast<K>(x_d[i]);
						return x;
					}
		};

	template<typename K>
		class Jacobi {
			public:
				aligned_vector<K> data;
				static inline Vector<K> solve(const Matrix<K>& A, const Vector<K>& b, K tol = 1e-10, int max_iter = 2000) {
					static_assert(std::is_floating_point<K>::value, "Jacobi solver requires floating-point type.");
					assert(A.rows == A.cols && "Matrix A must be square");
					assert(A.rows == b.size() && "Matrix/vector size mismatch");
					const size_t n = A.rows;
					Vector<K> x(n, K(0)); 
					Vector<K> x_new(n, K(0));

					using Simd = simd::SimdTraits<K, DefaultISA>;
					using reg = typename Simd::reg;
					const size_t simd_width = Simd::width;

					for (int iter = 0; iter < max_iter; ++iter) {
#pragma omp parallel for schedule(dynamic, 4) 
						for (size_t i = 0; i < n; ++i) {
							if (std::abs(A(i, i)) < 1e-10)
								throw std::runtime_error("Jacobi: division by near-zero on diagonal, matrix likely not diagonally dominant.");

							reg sum_vec = Simd::setzero();
							size_t j = 0;

							for (; j + simd_width <= n; j += simd_width) {
								reg a_vec = Simd::loadu(&A(i, j));
								reg x_vec = Simd::loadu(&x[j]);

								if (i >= j && i < j + simd_width) {
									std::array<K, simd_width> mask_arr;
									Simd::storeu(mask_arr.data(), a_vec);
									mask_arr[i - j] = K(0);
									a_vec = Simd::loadu(mask_arr.data());
								}

								sum_vec = Simd::fma(a_vec, x_vec, sum_vec); 
							}

							K sigma = Simd::horizontal_add(sum_vec);

							for (; j < n; ++j) {
								if (j != i)
									sigma += A(i, j) * x[j];
							}

							x_new[i] = (b[i] - sigma) / A(i, i);
						}

						K err = K(0);
#pragma omp parallel for reduction(+:err) schedule(dynamic)
						for (size_t i = 0; i < n; ++i) {
							K diff = x_new[i] - x[i];
							err += diff * diff;
						}
						err = std::sqrt(err);

						if (err < tol)
							break;

						x = x_new;
					}

					return x;
				}
		};

	template<typename K>
		class GaussSeidel {
			public:
				aligned_vector<K> data;
				static Vector<K> solve(const Matrix<K>& A, const Vector<K>& b, K tol = 1e-8, int max_iter = 2000);
		};

}

