#include "../framework/Assertions.hpp"
#include "../framework/TestRegistry.hpp"

#include "../../includes/Tensorium/Tensorium.hpp"

#include <array>

using namespace tensorium;

REGISTER_TEST("core.tensor.flatten", "Tensor flatten/unflatten consistency", []() {
    Tensor<float, 4> T({2, 3, 4, 5});
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                for (size_t l = 0; l < 5; ++l) {
                    const std::array<size_t, 4> idx = {i, j, k, l};
                    float value = static_cast<float>(T.flatten_index(idx));
                    T(idx) = value;
                    tensorium::tests::expect_near(T(idx), value, 1e-6, "tensor flatten");
                }
});

REGISTER_TEST("core.tensor.product", "Tensor products and contractions", []() {
    Tensor<float, 2> A({2, 2}), B({2, 2});
    A({0, 0}) = 1; A({0, 1}) = 2; A({1, 0}) = 3; A({1, 1}) = 4;
    B({0, 0}) = 10; B({0, 1}) = 20; B({1, 0}) = 30; B({1, 1}) = 40;

    auto C = mul_tensor(A, B);
    tensorium::tests::expect_near(C({0, 0, 0, 0}), 1 * 10, 1e-5, "tensor mul");
    tensorium::tests::expect_near(C({1, 1, 1, 1}), 4 * 40, 1e-5, "tensor mul");

    Tensor<float, 3> T3({2, 2, 2});
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            for (size_t k = 0; k < 2; ++k)
                T3({i, j, k}) = float(i + j + k);

    auto contracted = T3.contract_tensor<1, 2>();
    for (size_t i = 0; i < 2; ++i) {
        float expected = T3({i, 0, 0}) + T3({i, 1, 1});
        tensorium::tests::expect_near(contracted({i}), expected, 1e-5, "tensor contract");
    }
});

