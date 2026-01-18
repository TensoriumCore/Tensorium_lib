#include "../framework/Assertions.hpp"
#include "../framework/TestRegistry.hpp"

#include "../../includes/Tensorium/Tensorium.hpp"

#include <cmath>

using namespace tensorium;

REGISTER_TEST("core.derivate.axis0_order2", "Centered derivative along axis 0", []() {
    Derivate<double> input(8, 8);
    Derivate<double> output(8, 8);
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 8; ++j)
            input(i, j) = static_cast<double>(i * i) + 0.5 * static_cast<double>(j);
    centered_derivative(input, output, 0, 1.0);
    tensorium::tests::expect_near(output(4, 3), 8.0, 1e-9, "axis0 order2");
    tensorium::tests::expect_near(output(5, 5), 10.0, 1e-9, "axis0 order2 second point");
});

REGISTER_TEST("core.derivate.axis1_order4", "Fourth-order derivative along axis 1", []() {
    Derivate<double> input(10, 10);
    Derivate<double> output(10, 10);
    for (size_t i = 0; i < 10; ++i)
        for (size_t j = 0; j < 10; ++j)
            input(i, j) = std::pow(static_cast<double>(j), 4) + static_cast<double>(i);
    centered_derivative_order4(input, output, 1, 1.0);
    tensorium::tests::expect_near(output(5, 5), 4.0 * std::pow(5.0, 3), 1e-6, "axis1 order4");
});
