#pragma once

#include "../../../Core/Spectral.hpp"
#include "../Metric.hpp"
#include "BSSNGridSetupUtils.hpp"
#include <array>
#include <stdexcept>
namespace tensorium_RG {

namespace detail {
template <typename T> inline std::array<size_t, 3> spatial_indices(const tensorium::Vector<T> &X) {
    const size_t n = X.size();
    if (n == 3)
        return {0, 1, 2};
    if (n >= 3)
        return {n - 3, n - 2, n - 1};
    throw std::invalid_argument("Vector must contain at least three spatial components");
}

template <typename T>
inline void shift_axis(tensorium::Vector<T> &X, const std::array<size_t, 3> &idx, size_t axis, T delta) {
    X(idx[axis]) += delta;
}

template <typename T>
inline tensorium::Vector<T> shifted_copy(const tensorium::Vector<T> &X,
                                         const std::array<size_t, 3> &idx, T dx, T dy, T dz) {
    tensorium::Vector<T> Xs = X;
    Xs(idx[0]) += dx;
    Xs(idx[1]) += dy;
    Xs(idx[2]) += dz;
    return Xs;
}
} // namespace detail

template <typename T>
void spectral_derivative_1D(tensorium::Vector<T> &field, tensorium::Vector<T> &dfield, T dx) {
    using C = std::complex<T>;
    using FFT = tensorium::SpectralFFT<T>;

    tensorium::Vector<C> field_fft(field.size());
    for (size_t i = 0; i < field.size(); ++i)
        field_fft(i) = field(i);

    FFT::forward(field_fft);

    const T L = dx * field.size();
    for (size_t k = 0; k < field.size(); ++k) {
        int n = (k <= field.size() / 2) ? k : (int)k - (int)field.size();
        C   factor = C(0, 2.0 * M_PI * n / L);
        field_fft(k) *= factor;
    }

    FFT::backward(field_fft);
    for (size_t i = 0; i < field.size(); ++i)
        dfield(i) = field_fft(i).real();
}

template <typename T>
void spectral_partial_scalar_3D(const tensorium::Tensor<T, 3> &scalar_field, T dx, T dy, T dz,
                                tensorium::Tensor<T, 4> &grad_out) {
    using namespace tensorium;
    using C = std::complex<T>;
    using FFT = SpectralFFT<T>;

    const auto  &shape = scalar_field.shape();
    const size_t NX = shape[0], NY = shape[1], NZ = shape[2];

    grad_out.resize({NX, NY, NZ, 3});

    Tensor<C, 3> field_cplx({NX, NY, NZ});
    for (size_t i = 0; i < NX; ++i)
        for (size_t j = 0; j < NY; ++j)
            for (size_t k = 0; k < NZ; ++k)
                field_cplx(i, j, k) = C(scalar_field(i, j, k), 0.0);

    FFT::forward_3D(field_cplx);

    for (size_t i = 0; i < NX; ++i) {
        int ni = (i <= NX / 2) ? int(i) : int(i) - int(NX);
        T   kx = 2.0 * M_PI * ni / (NX * dx);

        for (size_t j = 0; j < NY; ++j) {
            int nj = (j <= NY / 2) ? int(j) : int(j) - int(NY);
            T   ky = 2.0 * M_PI * nj / (NY * dy);

            for (size_t k = 0; k < NZ; ++k) {
                int nk = (k <= NZ / 2) ? int(k) : int(k) - int(NZ);
                T   kz = 2.0 * M_PI * nk / (NZ * dz);

                C f_hat = field_cplx(i, j, k);

                field_cplx(i, j, k) = f_hat;

                grad_out(i, j, k, 0) = (f_hat * C(0, kx)).real();
                grad_out(i, j, k, 1) = (f_hat * C(0, ky)).real();
                grad_out(i, j, k, 2) = (f_hat * C(0, kz)).real();
            }
        }
    }

    for (int dim = 0; dim < 3; ++dim) {
        Tensor<C, 3> dfield_cplx({NX, NY, NZ});

        for (size_t i = 0; i < NX; ++i)
            for (size_t j = 0; j < NY; ++j)
                for (size_t k = 0; k < NZ; ++k)
                    dfield_cplx(i, j, k) = C(grad_out(i, j, k, dim), 0.0);

        FFT::backward_3D(dfield_cplx);

        for (size_t i = 0; i < NX; ++i)
            for (size_t j = 0; j < NY; ++j)
                for (size_t k = 0; k < NZ; ++k)
                    grad_out(i, j, k, dim) = dfield_cplx(i, j, k).real();
    }
}

template <typename T, typename TensorFunc>
tensorium::Tensor<T, 3> spectral_partial_tensor2(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                                                 TensorFunc &&func, size_t NX, size_t NY,
                                                 size_t NZ) {
    using namespace tensorium;
    using C = std::complex<T>;
    using FFT = SpectralFFT<T>;

    Tensor<T, 2> gamma = func(X);
    Tensor<T, 3> grad_gamma({3, 3, 3});

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            Tensor<T, 3> gamma_ij_grid =
                tensorium_RG::populate_tensor3D_component<T>(i, j, func, dx, dy, dz, NX, NY, NZ);

            Tensor<T, 4> grad_tmp;
            spectral_partial_scalar_3D(gamma_ij_grid, dx, dy, dz, grad_tmp);

            for (size_t k = 0; k < 3; ++k)
                grad_gamma(i, j, k) = grad_tmp(NX / 2, NY / 2, NZ / 2, k);
        }
    }

    return grad_gamma;
}

template <typename T, typename ScalarFunc>
tensorium::Vector<T> partial_scalar(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                                    ScalarFunc &&func) {
    tensorium::Vector<T>      grad(3);
    const auto             idx = detail::spatial_indices(X);
    const std::array<T, 3> spacings{dx, dy, dz};

    for (size_t axis = 0; axis < 3; ++axis) {
        const T h = spacings[axis];
        auto sample = [&](int step) {
            auto Xs = X;
            detail::shift_axis(Xs, idx, axis, step * h);
            return func(Xs);
        };

        T gm2 = sample(-2);
        T gm1 = sample(-1);
        T gp1 = sample(1);
        T gp2 = sample(2);
        grad(axis) = (-gp2 + 8 * gp1 - 8 * gm1 + gm2) / (12 * h);
    }

    return grad;
}

template <typename T, typename VectorFunc>
tensorium::Tensor<T, 2> partial_vector(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                                       VectorFunc &&func) {
    tensorium::Tensor<T, 2> result({3, 3});
    const auto              idx = detail::spatial_indices(X);
    const std::array<T, 3>  spacings{dx, dy, dz};

    for (size_t axis = 0; axis < 3; ++axis) {
        const T h = spacings[axis];
        auto sample = [&](int step) {
            auto Xs = X;
            detail::shift_axis(Xs, idx, axis, step * h);
            return func(Xs);
        };

        tensorium::Vector<T> Vm2 = sample(-2);
        tensorium::Vector<T> Vm1 = sample(-1);
        tensorium::Vector<T> Vp1 = sample(1);
        tensorium::Vector<T> Vp2 = sample(2);

        for (int i = 0; i < 3; ++i)
            result(i, axis) = (-Vp2(i) + 8 * Vp1(i) - 8 * Vm1(i) + Vm2(i)) / (12 * h);
    }

    return result;
}

template <typename T, typename TensorFunc>
void compute_partial_derivatives_tensor2D(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                                          TensorFunc &&func, tensorium::Tensor<T, 3> &out) {
    out.resize(3, 3, 3);

    const auto idx = detail::spatial_indices(X);

    auto shifted = [&](T dx_, T dy_, T dz_) {
        auto                       Xs = detail::shifted_copy(X, idx, dx_, dy_, dz_);
        tensorium::Tensor<T, 2>    out_tensor({3, 3});
        func(Xs, out_tensor);
        return out_tensor;
    };

    for (size_t a = 0; a < 3; ++a) {
        for (size_t b = 0; b < 3; ++b) {
            // ∂₀ = ∂/∂x
            auto gm2 = shifted(-2 * dx, 0, 0);
            auto gm1 = shifted(-dx, 0, 0);
            auto gp1 = shifted(dx, 0, 0);
            auto gp2 = shifted(2 * dx, 0, 0);
            out(0, a, b) = (-gp2(a, b) + 8 * gp1(a, b) - 8 * gm1(a, b) + gm2(a, b)) / (12 * dx);

            // ∂₁ = ∂/∂y
            gm2 = shifted(0, -2 * dy, 0);
            gm1 = shifted(0, -dy, 0);
            gp1 = shifted(0, dy, 0);
            gp2 = shifted(0, 2 * dy, 0);
            out(1, a, b) = (-gp2(a, b) + 8 * gp1(a, b) - 8 * gm1(a, b) + gm2(a, b)) / (12 * dy);

            // ∂₂ = ∂/∂z
            gm2 = shifted(0, 0, -2 * dz);
            gm1 = shifted(0, 0, -dz);
            gp1 = shifted(0, 0, dz);
            gp2 = shifted(0, 0, 2 * dz);
            out(2, a, b) = (-gp2(a, b) + 8 * gp1(a, b) - 8 * gm1(a, b) + gm2(a, b)) / (12 * dz);
        }
    }
}

template <typename T, typename TensorFunc>
void compute_second_derivatives_tensor2D(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                                         TensorFunc &&func, tensorium::Tensor<T, 4> &out) {
    out = tensorium::Tensor<T, 4>({3, 3, 3, 3});

    const auto idx = detail::spatial_indices(X);

    auto shifted = [&](T dx_, T dy_, T dz_) {
        auto                    Xs = detail::shifted_copy(X, idx, dx_, dy_, dz_);
        tensorium::Tensor<T, 2> out_tensor({3, 3});
        func(Xs, out_tensor);
        return out_tensor;
    };

    for (size_t a = 0; a < 3; ++a) {
        for (size_t b = 0; b < 3; ++b) {
            // ∂²/∂x²
            auto gm2 = shifted(-2 * dx, 0, 0);
            auto gm1 = shifted(-dx, 0, 0);
            auto g0 = shifted(0, 0, 0);
            auto gp1 = shifted(dx, 0, 0);
            auto gp2 = shifted(2 * dx, 0, 0);
            out(0, a, b, 0) =
                (-gp2(a, b) + 16 * gp1(a, b) - 30 * g0(a, b) + 16 * gm1(a, b) - gm2(a, b)) /
                (12 * dx * dx);

            // ∂²/∂y²
            gm2 = shifted(0, -2 * dy, 0);
            gm1 = shifted(0, -dy, 0);
            g0 = shifted(0, 0, 0);
            gp1 = shifted(0, dy, 0);
            gp2 = shifted(0, 2 * dy, 0);
            out(1, a, b, 1) =
                (-gp2(a, b) + 16 * gp1(a, b) - 30 * g0(a, b) + 16 * gm1(a, b) - gm2(a, b)) /
                (12 * dy * dy);

            // ∂²/∂z²
            gm2 = shifted(0, 0, -2 * dz);
            gm1 = shifted(0, 0, -dz);
            g0 = shifted(0, 0, 0);
            gp1 = shifted(0, 0, dz);
            gp2 = shifted(0, 0, 2 * dz);
            out(2, a, b, 2) =
                (-gp2(a, b) + 16 * gp1(a, b) - 30 * g0(a, b) + 16 * gm1(a, b) - gm2(a, b)) /
                (12 * dz * dz);
        }
    }
}

template <typename T, typename VectorFunc>
void compute_partial_derivatives_vector(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                                        VectorFunc &&func, tensorium::Tensor<T, 2> &out) {
    out.resize(3, 3);

    const auto idx = detail::spatial_indices(X);

    auto shifted = [&](T dx_, T dy_, T dz_) {
        auto                Xs = detail::shifted_copy(X, idx, dx_, dy_, dz_);
        tensorium::Vector<T> Vout(3);
        func(Xs, Vout);
        return Vout;
    };

    for (size_t i = 0; i < 3; ++i) {
        auto gm2 = shifted(-2 * dx, 0, 0);
        auto gm1 = shifted(-dx, 0, 0);
        auto gp1 = shifted(dx, 0, 0);
        auto gp2 = shifted(2 * dx, 0, 0);
        out(0, i) = (-gp2(i) + 8 * gp1(i) - 8 * gm1(i) + gm2(i)) / (12 * dx);

        gm2 = shifted(0, -2 * dy, 0);
        gm1 = shifted(0, -dy, 0);
        gp1 = shifted(0, dy, 0);
        gp2 = shifted(0, 2 * dy, 0);
        out(1, i) = (-gp2(i) + 8 * gp1(i) - 8 * gm1(i) + gm2(i)) / (12 * dy);

        gm2 = shifted(0, 0, -2 * dz);
        gm1 = shifted(0, 0, -dz);
        gp1 = shifted(0, 0, dz);
        gp2 = shifted(0, 0, 2 * dz);
        out(2, i) = (-gp2(i) + 8 * gp1(i) - 8 * gm1(i) + gm2(i)) / (12 * dz);
    }
}

template <typename T, typename ScalarFunc>
void compute_partial_derivatives_scalar(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                                        ScalarFunc &&func, tensorium::Vector<T> &out) {

    out.resize(3);

    const auto idx = detail::spatial_indices(X);

    auto shifted = [&](T dx_, T dy_, T dz_) {
        auto Xs = detail::shifted_copy(X, idx, dx_, dy_, dz_);
        return func(Xs);
    };

    // ∂/∂x
    T gm2 = shifted(-2 * dx, 0, 0);
    T gm1 = shifted(-dx, 0, 0);
    T gp1 = shifted(dx, 0, 0);
    T gp2 = shifted(2 * dx, 0, 0);
    out[0] = (-gp2 + 8 * gp1 - 8 * gm1 + gm2) / (12 * dx);

    // ∂/∂y
    gm2 = shifted(0, -2 * dy, 0);
    gm1 = shifted(0, -dy, 0);
    gp1 = shifted(0, dy, 0);
    gp2 = shifted(0, 2 * dy, 0);
    out[1] = (-gp2 + 8 * gp1 - 8 * gm1 + gm2) / (12 * dy);

    // ∂/∂z
    gm2 = shifted(0, 0, -2 * dz);
    gm1 = shifted(0, 0, -dz);
    gp1 = shifted(0, 0, dz);
    gp2 = shifted(0, 0, 2 * dz);
    out[2] = (-gp2 + 8 * gp1 - 8 * gm1 + gm2) / (12 * dz);
}

template <typename T>
tensorium::Tensor<T, 2>
compute_dt_gamma_from_beta(const tensorium::Vector<T> &beta_cov,
                           const tensorium::Tensor<T, 2> &partial_beta_cov,
                           const tensorium::Tensor<T, 3> &christoffel) {

    tensorium::Tensor<T, 2> dtg({3, 3});
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j) {
            T Di_bj = partial_beta_cov(i, j);
            T Dj_bi = partial_beta_cov(j, i);
            for (size_t k = 0; k < 3; ++k) {
                Di_bj -= christoffel(k, i, j) * beta_cov(k);
                Dj_bi -= christoffel(k, j, i) * beta_cov(k);
            }
            dtg(i, j) = Di_bj + Dj_bi;
        }
    return dtg;
}

template <typename T>
inline void compute_partial_derivatives_vector3D(const tensorium::Tensor<T, 4> &vec_field, size_t i,
                                                 size_t j, size_t k, T dx, T dy, T dz,
                                                 tensorium::Tensor<T, 2> &dvec_out) {
    dvec_out.resize(3, 3);

    auto get = [&](int ii, int jj, int kk, int a) -> T { return vec_field(ii, jj, kk, a); };

    const auto  &shape = vec_field.shape();
    const size_t NX = shape[0], NY = shape[1], NZ = shape[2];

    for (size_t a = 0; a < 3; ++a) {
        if (i >= 2 && i + 2 < NX)
            dvec_out(0, a) = (-get(i + 2, j, k, a) + 8 * get(i + 1, j, k, a) -
                              8 * get(i - 1, j, k, a) + get(i - 2, j, k, a)) /
                             (12 * dx);
        else if (i > 0 && i + 1 < NX)
            dvec_out(0, a) = (get(i + 1, j, k, a) - get(i - 1, j, k, a)) / (2 * dx);
        else
            dvec_out(0, a) = T(0);

        if (j >= 2 && j + 2 < NY)
            dvec_out(1, a) = (-get(i, j + 2, k, a) + 8 * get(i, j + 1, k, a) -
                              8 * get(i, j - 1, k, a) + get(i, j - 2, k, a)) /
                             (12 * dy);
        else if (j > 0 && j + 1 < NY)
            dvec_out(1, a) = (get(i, j + 1, k, a) - get(i, j - 1, k, a)) / (2 * dy);
        else
            dvec_out(1, a) = T(0);

        if (k >= 2 && k + 2 < NZ)
            dvec_out(2, a) = (-get(i, j, k + 2, a) + 8 * get(i, j, k + 1, a) -
                              8 * get(i, j, k - 1, a) + get(i, j, k - 2, a)) /
                             (12 * dz);
        else if (k > 0 && k + 1 < NZ)
            dvec_out(2, a) = (get(i, j, k + 1, a) - get(i, j, k - 1, a)) / (2 * dz);
        else
            dvec_out(2, a) = T(0);
    }
}

template <typename T, typename ScalarFunc>
inline void compute_second_derivatives_scalar(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                                              ScalarFunc &&func, tensorium::Tensor<T, 2> &out) {
    out.resize(3, 3);

    const auto idx = detail::spatial_indices(X);

    auto shifted = [&](T dx_, T dy_, T dz_) {
        auto Xs = detail::shifted_copy(X, idx, dx_, dy_, dz_);
        return func(Xs);
    };

    // ∂²/∂x²
    {
        T gm2 = shifted(-2 * dx, 0, 0);
        T gm1 = shifted(-dx, 0, 0);
        T g0 = shifted(0, 0, 0);
        T gp1 = shifted(dx, 0, 0);
        T gp2 = shifted(2 * dx, 0, 0);
        out(0, 0) = (-gp2 + 16 * gp1 - 30 * g0 + 16 * gm1 - gm2) / (12 * dx * dx);
    }

    // ∂²/∂y²
    {
        T gm2 = shifted(0, -2 * dy, 0);
        T gm1 = shifted(0, -dy, 0);
        T g0 = shifted(0, 0, 0);
        T gp1 = shifted(0, dy, 0);
        T gp2 = shifted(0, 2 * dy, 0);
        out(1, 1) = (-gp2 + 16 * gp1 - 30 * g0 + 16 * gm1 - gm2) / (12 * dy * dy);
    }

    // ∂²/∂z²
    {
        T gm2 = shifted(0, 0, -2 * dz);
        T gm1 = shifted(0, 0, -dz);
        T g0 = shifted(0, 0, 0);
        T gp1 = shifted(0, 0, dz);
        T gp2 = shifted(0, 0, 2 * dz);
        out(2, 2) = (-gp2 + 16 * gp1 - 30 * g0 + 16 * gm1 - gm2) / (12 * dz * dz);
    }

    // ∂²/∂x∂y
    {
        T pp = shifted(dx, dy, 0);
        T pm = shifted(dx, -dy, 0);
        T mp = shifted(-dx, dy, 0);
        T mm = shifted(-dx, -dy, 0);
        out(0, 1) = out(1, 0) = (pp - pm - mp + mm) / (4 * dx * dy);
    }

    // ∂²/∂x∂z
    {
        T pp = shifted(dx, 0, dz);
        T pm = shifted(dx, 0, -dz);
        T mp = shifted(-dx, 0, dz);
        T mm = shifted(-dx, 0, -dz);
        out(0, 2) = out(2, 0) = (pp - pm - mp + mm) / (4 * dx * dz);
    }

    // ∂²/∂y∂z
    {
        T pp = shifted(0, dy, dz);
        T pm = shifted(0, dy, -dz);
        T mp = shifted(0, -dy, dz);
        T mm = shifted(0, -dy, -dz);
        out(1, 2) = out(2, 1) = (pp - pm - mp + mm) / (4 * dy * dz);
    }
}

template <typename T, typename ScalarFunc>
tensorium::Tensor<T, 2> covariant_scalar_second(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                                                ScalarFunc                   &&func,
                                                const tensorium::Tensor<T, 3> &christoffel) {
    using namespace tensorium;

    Vector<T> grad(3);
    compute_partial_derivatives_scalar(X, dx, dy, dz, func, grad);

    Tensor<T, 2> hess({3, 3});
    compute_second_derivatives_scalar(X, dx, dy, dz, std::forward<ScalarFunc>(func), hess);

    Tensor<T, 2> result({3, 3});
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            result(i, j) = hess(i, j);
            for (int k = 0; k < 3; ++k)
                result(i, j) -=
                    christoffel(k, i, j) * grad(k); // D_i D_j φ = ∂_i ∂_j φ - Γ^k_{ij} ∂_k φ
        }
    return result;
}

template <typename T>
void compute_second_derivatives_scalar3D(const tensorium::Tensor<T, 3> &scalar_field, T dx, T dy,
                                         T dz, tensorium::Tensor<T, 5> &hessian_out) {
    const auto  &shape = scalar_field.shape();
    const size_t NX = shape[0], NY = shape[1], NZ = shape[2];

    hessian_out.resize(NX, NY, NZ, 3, 3);

    auto get = [&](int i, int j, int k) -> T {
        if (i < 0)
            i = 0;
        if (j < 0)
            j = 0;
        if (k < 0)
            k = 0;
        if (i >= NX)
            i = NX - 1;
        if (j >= NY)
            j = NY - 1;
        if (k >= NZ)
            k = NZ - 1;
        return scalar_field(i, j, k);
    };

    for (size_t i = 0; i < NX; ++i) {
        for (size_t j = 0; j < NY; ++j) {
            for (size_t k = 0; k < NZ; ++k) {

                auto d2f_xx = (-get(i + 2, j, k) + 16 * get(i + 1, j, k) - 30 * get(i, j, k) +
                               16 * get(i - 1, j, k) - get(i - 2, j, k)) /
                              (12 * dx * dx);
                auto d2f_yy = (-get(i, j + 2, k) + 16 * get(i, j + 1, k) - 30 * get(i, j, k) +
                               16 * get(i, j - 1, k) - get(i, j - 2, k)) /
                              (12 * dy * dy);
                auto d2f_zz = (-get(i, j, k + 2) + 16 * get(i, j, k + 1) - 30 * get(i, j, k) +
                               16 * get(i, j, k - 1) - get(i, j, k - 2)) /
                              (12 * dz * dz);

                auto d2f_xy = (get(i + 1, j + 1, k) - get(i + 1, j - 1, k) - get(i - 1, j + 1, k) +
                               get(i - 1, j - 1, k)) /
                              (4 * dx * dy);
                auto d2f_xz = (get(i + 1, j, k + 1) - get(i + 1, j, k - 1) - get(i - 1, j, k + 1) +
                               get(i - 1, j, k - 1)) /
                              (4 * dx * dz);
                auto d2f_yz = (get(i, j + 1, k + 1) - get(i, j + 1, k - 1) - get(i, j - 1, k + 1) +
                               get(i, j - 1, k - 1)) /
                              (4 * dy * dz);

                hessian_out(i, j, k, 0, 0) = d2f_xx;
                hessian_out(i, j, k, 1, 1) = d2f_yy;
                hessian_out(i, j, k, 2, 2) = d2f_zz;

                hessian_out(i, j, k, 0, 1) = hessian_out(i, j, k, 1, 0) = d2f_xy;
                hessian_out(i, j, k, 0, 2) = hessian_out(i, j, k, 2, 0) = d2f_xz;
                hessian_out(i, j, k, 1, 2) = hessian_out(i, j, k, 2, 1) = d2f_yz;
            }
        }
    }
}

template <typename T>
tensorium::Tensor<T, 5> covariant_scalar_second_3D(const tensorium::Tensor<T, 3> &chi,
                                                   const tensorium::Tensor<T, 5> &christoffel, T dx,
                                                   T dy, T dz) {
    const auto  &shape = chi.shape();
    const size_t NX = shape[0], NY = shape[1], NZ = shape[2];

    tensorium::Tensor<T, 5> result({NX, NY, NZ, 3, 3});

    tensorium::Vector<T>    grad(3);
    tensorium::Tensor<T, 2> hess({3, 3});

    for (size_t i = 0; i < NX; ++i)
        for (size_t j = 0; j < NY; ++j)
            for (size_t k = 0; k < NZ; ++k) {
                auto func = [&](const tensorium::Vector<T> &X) -> T { return chi(i, j, k); };

                compute_partial_derivatives_scalar({T(i), T(j), T(k)}, dx, dy, dz, func, grad);
                compute_second_derivatives_scalar({T(i), T(j), T(k)}, dx, dy, dz, func, hess);

                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) {
                        T val = hess(a, b);
                        for (int c = 0; c < 3; ++c)
                            val -= christoffel(i, j, k, c, a, b) * grad(c);
                        result(i, j, k, a, b) = val;
                    }
            }

    return result;
}

template <typename T, typename VectorFunc>
tensorium::Tensor<T, 2> covariant_vector(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                                         VectorFunc &&func, const tensorium::Tensor<T, 3> &Gamma) {
    auto dVi = partial_vector(X, dx, dy, dz, std::forward<VectorFunc>(func));

    tensorium::Tensor<T, 2> result({3, 3});

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            result(i, j) = dVi(i, j);
            for (int k = 0; k < 3; ++k)
                result(i, j) += Gamma(i, j, k) * func(X)(k);
        }

    return result;
}

template <typename T, typename TensorFunc>
tensorium::Tensor<T, 3> covariant_tensor2(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                                          TensorFunc &&func, const tensorium::Tensor<T, 3> &Gamma) {
    auto                    dTij = partial_tensor2(X, dx, dy, dz, std::forward<TensorFunc>(func));
    tensorium::Tensor<T, 2> Tij = func(X);
    tensorium::Tensor<T, 3> result({3, 3, 3});

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                result(i, j, k) = dTij(i, j, k);
                for (int l = 0; l < 3; ++l)
                    result(i, j, k) += -Gamma(l, k, i) * Tij(l, j) - Gamma(l, k, j) * Tij(i, l);
            }

    return result;
}

template <typename T, typename TensorFunc>
tensorium::Tensor<T, 3> partial_tensor2(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                                        TensorFunc &&func) {
    tensorium::Tensor<T, 3> result({3, 3, 3});
    const auto              idx = detail::spatial_indices(X);
    const std::array<T, 3>  spacings{dx, dy, dz};

    for (int axis = 0; axis < 3; ++axis) {
        const T h = spacings[axis];
        auto Xm = X;
        auto Xp = X;
        detail::shift_axis(Xm, idx, axis, -h);
        detail::shift_axis(Xp, idx, axis, h);

        tensorium::Tensor<T, 2> fm = func(Xm);
        tensorium::Tensor<T, 2> fp = func(Xp);

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result(i, j, axis) = (fp(i, j) - fm(i, j)) / (2 * h);
    }

    return result;
}

template <typename T, typename TensorFunc>
tensorium::Tensor<T, 4> covariant_tensor2_second(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                                                 TensorFunc                   &&func,
                                                 const tensorium::Tensor<T, 3> &Gamma) {
    auto cov1 = [&](const tensorium::Vector<T> &Y) {
        return covariant_tensor2<T>(Y, dx, dy, dz, func, Gamma);
    };

    const auto             idx = detail::spatial_indices(X);
    tensorium::Tensor<T, 4> result({3, 3, 3, 3}); // ∇_i ∇_j T_{kl}

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l) {
                    auto X_minus = X;
                    auto X_plus = X;
                    const T h = (j == 0) ? dx : (j == 1 ? dy : dz);
                    detail::shift_axis(X_minus, idx, j, -h);
                    detail::shift_axis(X_plus, idx, j, h);

                    T cov_minus = cov1(X_minus)(k, l, i);
                    T cov_plus = cov1(X_plus)(k, l, i);
                    result(i, j, k, l) = (cov_plus - cov_minus) / (2 * h); // ∂_j (∇_i T_kl)
                }

    return result;
}

} // namespace tensorium_RG
