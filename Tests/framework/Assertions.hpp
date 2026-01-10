#pragma once

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tensorium::tests {

inline void raise_failure(const std::string &message) {
    throw std::runtime_error(message);
}

#define TENSORIUM_TEST_ASSERT(COND)                                                                    \
    do {                                                                                               \
        if (!(COND)) {                                                                                 \
            std::ostringstream _tensorium_assert_ss;                                                   \
            _tensorium_assert_ss << "Assertion failed: " << #COND << " at " << __FILE__ << ':'      \
                                  << __LINE__;                                                        \
            ::tensorium::tests::raise_failure(_tensorium_assert_ss.str());                             \
        }                                                                                              \
    } while (0)

inline void expect_near(double value, double expected, double tol, const std::string &label) {
    if (std::abs(value - expected) > tol) {
        std::ostringstream oss;
        oss << label << " expected " << expected << " ± " << tol << " but got " << value;
        raise_failure(oss.str());
    }
}

inline void expect_le(double value, double bound, const std::string &label) {
    if (value > bound) {
        std::ostringstream oss;
        oss << label << " expected <= " << bound << " but got " << value;
        raise_failure(oss.str());
    }
}

} // namespace tensorium::tests

