#include "../framework/Assertions.hpp"
#include "../framework/TestRegistry.hpp"

#include "../../includes/Tensorium/Tensorium.hpp"

#include <complex>

using namespace tensorium;

REGISTER_TEST("core.spectral.fft_roundtrip_1d", "1D FFT inverse matches input", []() {
    using Complex = std::complex<double>;
    Vector<Complex> data = {Complex(1.0, 0.5), Complex(-2.0, 1.0),
                             Complex(0.25, -0.75), Complex(3.0, 2.0),
                             Complex(-1.5, 0.0), Complex(0.0, 1.5),
                             Complex(2.5, -1.0), Complex(-0.5, 0.25)};
    Vector<Complex> original = data;
    SpectralFFT<double>::forward(data);
    SpectralFFT<double>::backward(data);
    for (size_t i = 0; i < data.size(); ++i) {
        tensorium::tests::expect_near(std::real(data[i]), std::real(original[i]), 1e-9, "fft real");
        tensorium::tests::expect_near(std::imag(data[i]), std::imag(original[i]), 1e-9, "fft imag");
    }
});

REGISTER_TEST("core.spectral.fft_roundtrip_3d", "3D FFT inverse matches input", []() {
    using Complex = std::complex<double>;
    Tensor<Complex, 3> tensor({4, 4, 4});
    Tensor<Complex, 3> reference({4, 4, 4});
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            for (size_t k = 0; k < 4; ++k) {
                Complex value(std::sin(0.5 * i) + j, std::cos(0.25 * k) - i);
                tensor({i, j, k}) = value;
                reference({i, j, k}) = value;
            }
    SpectralFFT<double>::forward_3D(tensor);
    SpectralFFT<double>::backward_3D(tensor);
    const double scale = 4.0 * 4.0 * 4.0;
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            for (size_t k = 0; k < 4; ++k) {
                auto idx = std::array<size_t, 3>{i, j, k};
                auto value = tensor(idx) * scale;
                tensorium::tests::expect_near(std::real(value), std::real(reference(idx)), 1e-6,
                                              "fft3d real");
                tensorium::tests::expect_near(std::imag(value), std::imag(reference(idx)), 1e-6,
                                              "fft3d imag");
            }
});
