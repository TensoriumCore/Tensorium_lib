#include "../framework/Assertions.hpp"
#include "../framework/TestRegistry.hpp"

#include "../../includes/Tensorium/Tensorium.hpp"
#include "../../includes/Tensorium/Backend/SIMD/SIMD.hpp"

#include <cstdint>

using namespace tensorium;

REGISTER_TEST("core.allocator.aligned_vector", "aligned_vector respects ALIGN", []() {
    aligned_vector<float> vec(128);
    constexpr std::size_t alignment = simd::SimdTraits<float, DefaultISA>::alignment;
    std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(vec.data());
    TENSORIUM_TEST_ASSERT(addr % alignment == 0);
    for (size_t i = 0; i < vec.size(); ++i)
        vec[i] = static_cast<float>(i);
    tensorium::tests::expect_near(vec[64], 64.0f, 1e-6f, "aligned data access");
});

REGISTER_TEST("core.allocator.reallocation", "Reallocation preserves alignment", []() {
    aligned_vector<double> vec(16, 1.0);
    constexpr std::size_t alignment = simd::SimdTraits<double, DefaultISA>::alignment;
    std::uintptr_t addr1 = reinterpret_cast<std::uintptr_t>(vec.data());
    vec.resize(1024);
    std::uintptr_t addr2 = reinterpret_cast<std::uintptr_t>(vec.data());
    TENSORIUM_TEST_ASSERT(addr2 % alignment == 0);
    tensorium::tests::expect_near(vec[0], 1.0, 1e-12, "initial preserved");
    vec[512] = 42.0;
    tensorium::tests::expect_near(vec[512], 42.0, 1e-12, "write after resize");
    TENSORIUM_TEST_ASSERT(addr1 != addr2);
});
