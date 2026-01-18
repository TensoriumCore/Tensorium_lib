#include "../framework/Assertions.hpp"
#include "../framework/TestRegistry.hpp"

#include "../../includes/Tensorium/Tensorium.hpp"

#include <array>
#include <cmath>

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

REGISTER_TEST("core.tensor.default_init", "Default-constructed tensors have zero size", []() {
    Tensor<double, 2> tensor_default;
    auto dims = tensor_default.shape();
    TENSORIUM_TEST_ASSERT(dims[0] == 0 && dims[1] == 0);
    TENSORIUM_TEST_ASSERT(tensor_default.total_size == 0);
    TENSORIUM_TEST_ASSERT(tensor_default.data.empty());
});

REGISTER_TEST("core.tensor.derivate_order4", "Centered order-4 derivative runs without recursion", []() {
    Derivate<double> input(8, 8);
    Derivate<double> output(8, 8);
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 8; ++j)
            input(i, j) = std::pow(static_cast<double>(j), 4) + static_cast<double>(i);
    centered_derivative_order4(input, output, 1, 1.0);
    double expected = 4.0 * std::pow(3.0, 3);
    tensorium::tests::expect_near(output(3, 3), expected, 1e-6, "derivate order4");
});

REGISTER_TEST("core.tensor.tensor_product_shape", "Tensor product shape correctness", []() {
    Tensor<float, 1> v({3});
    Tensor<float, 2> m({2, 2});
    for (size_t i = 0; i < 3; ++i)
        v({i}) = static_cast<float>(i + 1);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            m({i, j}) = static_cast<float>(i + j);
    auto prod = mul_tensor(v, m);
    tensorium::tests::expect_near(prod({0, 0, 0}), v({0}) * m({0, 0}), 1e-6, "tensor product value");
    tensorium::tests::expect_near(prod({2, 1, 1}), v({2}) * m({1, 1}), 1e-6, "tensor product value2");
});

REGISTER_TEST("core.tensor.transpose", "Tensor transpose for rank-2 tensor", []() {
    Tensor<double, 2> T({3, 4});
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            T({i, j}) = static_cast<double>(i * 10 + j);
    auto transposed = T.transpose_simd();
    TENSORIUM_TEST_ASSERT(transposed.dimensions[0] == 4 && transposed.dimensions[1] == 3);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            tensorium::tests::expect_near(transposed({j, i}), T({i, j}), 1e-12, "tensor transpose");
});

