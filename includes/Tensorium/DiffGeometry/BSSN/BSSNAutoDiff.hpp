#pragma once

#include "../Metric.hpp"
#include "BSSNDerivatives.hpp"
#include "../../Core/Tensor.hpp"

namespace tensorium {
    template<typename T>
    struct TensorTraits;

    template<typename T, size_t Rank>
    struct TensorTraits<Tensor<T, Rank>> {
        static constexpr size_t rank = Rank;
        using value_type = T;
    };

    template<typename T>
    struct TensorTraits {
        static constexpr size_t rank = 0;
        using value_type = T;
    };

    template<typename T>
    struct TensorTraits<Vector<T>> {
        static constexpr size_t rank = 1;
        using value_type = T;
    };
}

namespace tensorium_RG {

    enum class DiffMode { PARTIAL, COV, COV2 };

    // ===================== RANK 0 (SCALAR) =====================
    // Return type : Tensor<T,1> (i.e. un vecteur gradient)
    template<typename T, typename ScalarFunc>
    tensorium::Tensor<T, 1> autodiff_rank0(
        const tensorium::Vector<T>& X,
        T dx, T dy, T dz,
        ScalarFunc&& func,
        DiffMode mode,
        const tensorium::Tensor<T, 3>& christoffel)
    {
        using namespace tensorium;
        if (mode == DiffMode::PARTIAL) {
            return partial_scalar(X, dx, dy, dz, std::forward<ScalarFunc>(func));
        }
        else if (mode == DiffMode::COV2) {
            return covariant_scalar_second(X, dx, dy, dz, std::forward<ScalarFunc>(func), christoffel);
        }
        else {
            throw std::invalid_argument("autodiff_rank0: uniquement PARTIAL ou COV2 autorisés pour un champ scalaire");
        }
    }

    // ===================== RANK 1 (VECTOR) =====================
    // Return type : Tensor<T,2> (i.e. une matrice dérivée covariante ou jacobienne)
    template<typename T, typename VectorFunc>
    tensorium::Tensor<T, 2> autodiff_rank1(
        const tensorium::Vector<T>& X,
        T dx, T dy, T dz,
        VectorFunc&& func,
        DiffMode mode,
        const tensorium::Tensor<T, 3>& christoffel)
    {
        using namespace tensorium;
        if (mode == DiffMode::PARTIAL) {
            return partial_vector(X, dx, dy, dz, std::forward<VectorFunc>(func));
        }
        else if (mode == DiffMode::COV) {
            return covariant_vector(X, dx, dy, dz, std::forward<VectorFunc>(func), christoffel);
        }
        else {
            throw std::invalid_argument("autodiff_rank1: uniquement PARTIAL ou COV autorisés pour un champ vectoriel");
        }
    }

    // ============== RANK 2 (TENSOR 2D) : PREMIÈRE DÉRIVÉE SEULEMENT ==============
    // Nous n'implémentons ici que PARTIAL et COV. Si on demande COV2, on lève une exception.
    // Return type : Tensor<T,3> (i.e. un tenseur à 3 indices : ∂_k T_{ij} ou ∇_k T_{ij})
    template<typename T, typename TensorFunc>
    tensorium::Tensor<T, 3> autodiff_rank2_first(
        const tensorium::Vector<T>& X,
        T dx, T dy, T dz,
        TensorFunc&& func,
        DiffMode mode,
        const tensorium::Tensor<T, 3>& christoffel)
    {
        using namespace tensorium;
        if (mode == DiffMode::PARTIAL) {
            return partial_tensor2(X, dx, dy, dz, std::forward<TensorFunc>(func));
        }
        else if (mode == DiffMode::COV) {
            return covariant_tensor2(X, dx, dy, dz, std::forward<TensorFunc>(func), christoffel);
        }
        else {
            throw std::invalid_argument("autodiff_rank2_first: uniquement PARTIAL ou COV autorisés pour un tenseur d'ordre 2");
        }
    }

    // ===================== WRAPPER GÉNÉRIQUE =====================
    template<typename T, typename FieldFunc>
    auto autodiff(
        const tensorium::Vector<T>& X,
        T dx, T dy, T dz,
        FieldFunc&& func,
        DiffMode mode,
        const tensorium::Tensor<T, 3>& christoffel = {})
    {
        using namespace tensorium;
        auto sample = func(X);
        constexpr size_t rank = TensorTraits<std::decay_t<decltype(sample)>>::rank;

        if constexpr (rank == 0) {
            return autodiff_rank0(X, dx, dy, dz, std::forward<FieldFunc>(func), mode, christoffel);
        }
        else if constexpr (rank == 1) {
            return autodiff_rank1(X, dx, dy, dz, std::forward<FieldFunc>(func), mode, christoffel);
        }
        else if constexpr (rank == 2) {
            // On autorise ici PARTIAL et COV uniquement pour un champ « Tensor, rank2 ».
            // Si on veut une dérivée covariante seconde d'un tenseur 2D, l'utilisateur
            // devra appeler explicitement la fonction covariant_tensor2_second(...) par lui-même.
            return autodiff_rank2_first(X, dx, dy, dz, std::forward<FieldFunc>(func), mode, christoffel);
        }
        else {
            static_assert(rank <= 2, "autodiff: seuls les rangs 0 (scalaire), 1 (vecteur) et 2 (tenseur2) sont supportés.");
        }
    }

} // namespace tensorium_RG
