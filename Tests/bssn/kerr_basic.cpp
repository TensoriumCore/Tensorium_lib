#include "../framework/Assertions.hpp"
#include "../framework/TestRegistry.hpp"

#include "../../includes/Tensorium/Physics/DiffGeometry/Metric.hpp"

#include <cmath>

REGISTER_TEST("bssn.metric.kerr_schild", "Kerr-Schild metric produces finite ADM split", []() {
    tensorium_RG::Metric<double> metric("kerr_schild", 1.0, 0.5);

    tensorium::Vector<double> X(4);
    X(0) = 0.0;
    X(1) = 5.0;
    X(2) = 0.25;
    X(3) = -0.1;

    tensorium::Tensor<double, 2> g({4, 4});
    metric(X, g);

    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            TENSORIUM_TEST_ASSERT(std::isfinite(g(i, j)));

    TENSORIUM_TEST_ASSERT(g(0, 0) < -0.5);

    double                    alpha = 0.0;
    tensorium::Vector<double> beta(3);
    tensorium::Tensor<double, 2> gamma({3, 3});
    metric.BSSN(X, alpha, beta, gamma);

    TENSORIUM_TEST_ASSERT(alpha > 0.0);
    for (size_t i = 0; i < 3; ++i)
        TENSORIUM_TEST_ASSERT(std::isfinite(beta(i)));
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            TENSORIUM_TEST_ASSERT(std::isfinite(gamma(i, j)));
});
