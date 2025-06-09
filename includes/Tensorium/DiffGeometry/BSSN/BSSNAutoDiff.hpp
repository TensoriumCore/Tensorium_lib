#pragma once

#include "../../Core/Tensor.hpp"
#include "../Metric.hpp"
#include "BSSNDerivatives.hpp"

namespace tensorium {
template <typename T> struct TensorTraits;

template <typename T, size_t Rank> struct TensorTraits<Tensor<T, Rank>> {
    static constexpr size_t rank = Rank;
    using value_type = T;
};

template <typename T> struct TensorTraits {
    static constexpr size_t rank = 0;
    using value_type = T;
};

template <typename T> struct TensorTraits<Vector<T>> {
    static constexpr size_t rank = 1;
    using value_type = T;
};
} // namespace tensorium

namespace tensorium_RG {

enum class DiffMode { PARTIAL, COV, COV2, SPEC };

template <typename T, typename ScalarFunc>
tensorium::Vector<T> covariant_scalar(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                                      ScalarFunc                   &&func,
                                      const tensorium::Tensor<T, 3> &christoffel) {
    tensorium::Vector<T> grad = partial_scalar(X, dx, dy, dz, std::forward<ScalarFunc>(func));
    return grad;
}

template <typename T, typename ScalarFunc>
tensorium::Tensor<T, 2> autodiff_scalar_second(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                                               ScalarFunc                   &&func,
                                               const tensorium::Tensor<T, 3> &christoffel) {
    return covariant_scalar_second(X, dx, dy, dz, std::forward<ScalarFunc>(func), christoffel);
}

template <typename T, typename ScalarFunc>
tensorium::Vector<T> autodiff_rank0(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                                    ScalarFunc &&func, DiffMode mode,
                                    const tensorium::Tensor<T, 3> &christoffel) {
    using namespace tensorium;
    if (mode == DiffMode::PARTIAL) {
        return partial_scalar(X, dx, dy, dz, std::forward<ScalarFunc>(func));
    } else if (mode == DiffMode::COV) {
        return covariant_scalar(X, dx, dy, dz, std::forward<ScalarFunc>(func), christoffel);
    } else {
        throw std::invalid_argument("autodiff_rank0: PARTIAL ou COV uniquement. Utiliser "
                                    "`autodiff_scalar_second` pour COV2.");
    }
}

template <typename T, typename VectorFunc>
tensorium::Tensor<T, 2> autodiff_rank1(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                                       VectorFunc &&func, DiffMode mode,
                                       const tensorium::Tensor<T, 3> &christoffel) {
    using namespace tensorium;
    if (mode == DiffMode::PARTIAL) {
        return partial_vector(X, dx, dy, dz, std::forward<VectorFunc>(func));
    } else if (mode == DiffMode::COV) {
        return covariant_vector(X, dx, dy, dz, std::forward<VectorFunc>(func), christoffel);
    } else {
        throw std::invalid_argument(
            "autodiff_rank1: uniquement PARTIAL ou COV autorisés pour un champ vectoriel");
    }
}

template <typename T, typename TensorFunc>
tensorium::Tensor<T, 3> autodiff_rank2_first(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                                             TensorFunc &&func, DiffMode mode,
                                             const tensorium::Tensor<T, 3> &christoffel) {
    using namespace tensorium;
    if (mode == DiffMode::PARTIAL) {
        return partial_tensor2(X, dx, dy, dz, std::forward<TensorFunc>(func));
    } else if (mode == DiffMode::COV) {
        return covariant_tensor2(X, dx, dy, dz, std::forward<TensorFunc>(func), christoffel);
    } else if (mode == DiffMode::SPEC) {
        return spectral_partial_tensor2<T>(X, dx, dy, dz, std::forward<TensorFunc>(func), 64, 64,
                                           64);
    } else {
        throw std::invalid_argument(
            "autodiff_rank2_first: uniquement PARTIAL ou COV autorisés pour un tenseur d'ordre 2");
    }
}

template <typename T, typename FieldFunc>
auto autodiff(const tensorium::Vector<T> &X, T dx, T dy, T dz, FieldFunc &&func, DiffMode mode,
              const tensorium::Tensor<T, 3> &christoffel = {}) {
    using namespace tensorium;
    auto sample = func(X);

    constexpr size_t rank = TensorTraits<std::decay_t<decltype(sample)>>::rank;

    if constexpr (rank == 0) {
        if (mode == DiffMode::COV2)
            return autodiff_scalar_second(X, dx, dy, dz, std::forward<FieldFunc>(func),
                                          christoffel); // <- nouveau chemin
        else
            return autodiff_rank0(X, dx, dy, dz, std::forward<FieldFunc>(func), mode, christoffel);
    } else if constexpr (rank == 1) {
        return autodiff_rank1(X, dx, dy, dz, std::forward<FieldFunc>(func), mode, christoffel);
    } else if constexpr (rank == 2) {
        return autodiff_rank2_first(X, dx, dy, dz, std::forward<FieldFunc>(func), mode,
                                    christoffel);
    } else {
        static_assert(
            rank <= 2,
            "autodiff: seuls les rangs 0 (scalaire), 1 (vecteur) et 2 (tenseur2) sont supportés.");
    }
}

} // namespace tensorium_RG
