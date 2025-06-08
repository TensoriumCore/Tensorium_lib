#pragma once
#include "../../Core/Tensor.hpp"
#include "../../Core/Vector.hpp"

namespace tensorium_RG {

	template<typename T, typename TensorFunc>
		tensorium::Tensor<T, 3> populate_tensor3D_component(
				size_t i, size_t j,
				TensorFunc&& func,
				T dx, T dy, T dz,
				size_t NX, size_t NY, size_t NZ)
		{
			using namespace tensorium;
			Tensor<T, 3> output({NX, NY, NZ});

			for (size_t x = 0; x < NX; ++x) {
				for (size_t y = 0; y < NY; ++y) {
					for (size_t z = 0; z < NZ; ++z) {
						Vector<T> X(4); 
						X(0) = 0.0; 
						X(1) = x * dx;
						X(2) = y * dy;
						X(3) = z * dz;

						Tensor<T, 2> gamma = func(X);
						output(x, y, z) = gamma(i, j);
					}
				}
			}
			return output;
		}

}
