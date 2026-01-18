#pragma once

#include "../../../Core/Tensor.hpp"
#include "../../../Core/Vector.hpp"
#include "../Metric.hpp"
#include "BSSNDerivatives.hpp"

namespace tensorium_RG {

template <typename T> struct ChiContext {
    tensorium::Vector<T> X;
    T                    dx, dy, dz;
    const Metric<T>     &metric;

    // Contenu calculé
    T                       chi;
    tensorium::Vector<T>    grad_chi;
    tensorium::Tensor<T, 2> hessian_chi;

    static ChiContext<T> compute(const tensorium::Vector<T> &X_, T dx_, T dy_, T dz_,
                                 const tensorium::Tensor<T, 2> &g_phys,
                                 const tensorium::Tensor<T, 3> &d_g_phys,
                                 const Metric<T>               &metric_) {
        using namespace tensorium;

        ChiContext<T> ctx{.X = X_, .dx = dx_, .dy = dy_, .dz = dz_, .metric = metric_};

        ctx.chi = compute_conformal_factor(metric_, g_phys);

        auto scalar_func = [&](const Vector<T> &Xs) -> T {
            T            alpha;
            Vector<T>    beta(3);
            Tensor<T, 2> g_tmp({3, 3});
            metric_.BSSN(Xs, alpha, beta, g_tmp);
            return compute_conformal_factor(metric_, g_tmp);
        };

        ctx.grad_chi = partial_scalar(X_, dx_, dy_, dz_, scalar_func);
        compute_second_derivatives_scalar(X_, dx_, dy_, dz_, scalar_func, ctx.hessian_chi);

        return ctx;
    }
};

} // namespace tensorium_RG
