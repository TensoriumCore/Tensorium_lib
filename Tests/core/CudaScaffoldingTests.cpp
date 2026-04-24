#include "../framework/Assertions.hpp"
#include "../framework/TestRegistry.hpp"

#include "../../includes/Tensorium/Tensorium.hpp"

#include <string>

using namespace tensorium;

REGISTER_TEST("core.cuda.runtime_metadata", "CUDA runtime metadata is exposed coherently", []() {
    TENSORIUM_TEST_ASSERT(std::string(tensorium::backend::to_string(tensorium::backend::Kind::CPU))
                          == "cpu");
    TENSORIUM_TEST_ASSERT(std::string(tensorium::backend::to_string(tensorium::backend::Kind::CUDA))
                          == "cuda");
    TENSORIUM_TEST_ASSERT(std::string(
                              tensorium::backend::to_string(tensorium::backend::MemorySpace::Host))
                          == "host");
    TENSORIUM_TEST_ASSERT(std::string(
                              tensorium::backend::to_string(tensorium::backend::MemorySpace::Device))
                          == "device");

#ifdef TENSORIUM_CUDA
    TENSORIUM_TEST_ASSERT(std::string(tensorium::cuda::compiled_arch_list()) != "disabled");
#else
    TENSORIUM_TEST_ASSERT(std::string(tensorium::cuda::compiled_arch_list()) == "disabled");
#endif
});

REGISTER_TEST("core.cuda.vector_roundtrip", "VectorCUDA roundtrip preserves host values", []() {
#ifdef TENSORIUM_CUDA
    if (!tensorium::cuda::is_available())
        return;

    Vector<double> host{1.0, -2.5, 4.25, 8.0};
    tensorium::cuda::VectorCUDA<double> device(host);
    Vector<double> roundtrip = device.to_host();

    TENSORIUM_TEST_ASSERT(roundtrip.size() == host.size());
    for (size_t i = 0; i < host.size(); ++i)
        tensorium::tests::expect_near(roundtrip[i], host[i], 1e-12, "VectorCUDA roundtrip");
#endif
});

REGISTER_TEST("core.cuda.matrix_roundtrip", "MatrixCUDA roundtrip preserves host values", []() {
#ifdef TENSORIUM_CUDA
    if (!tensorium::cuda::is_available())
        return;

    Matrix<double> host(2, 3);
    host(0, 0) = 1.0;
    host(1, 0) = 2.0;
    host(0, 1) = 3.0;
    host(1, 1) = 4.0;
    host(0, 2) = 5.0;
    host(1, 2) = 6.0;

    tensorium::cuda::MatrixCUDA<double> device(host);
    Matrix<double> roundtrip = device.to_host();

    TENSORIUM_TEST_ASSERT(roundtrip.rows == host.rows);
    TENSORIUM_TEST_ASSERT(roundtrip.cols == host.cols);
    for (size_t j = 0; j < host.cols; ++j) {
        for (size_t i = 0; i < host.rows; ++i) {
            tensorium::tests::expect_near(roundtrip(i, j), host(i, j), 1e-12,
                                          "MatrixCUDA roundtrip");
        }
    }
#endif
});
