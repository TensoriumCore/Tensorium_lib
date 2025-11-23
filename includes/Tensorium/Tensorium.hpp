#pragma once

// --- BACKEND (Moteur) ---
#include "Backend/SIMD/SIMD.hpp"
#include "Backend/CPU_Kernels/GemmKernel_Optimized.hpp"

#ifdef TENSORIUM_USE_CUDA
    #include "Backend/CUDA/Core/MatrixCUDA.hpp"
#endif

#include "Core/Matrix.hpp"
#include "Core/Vector.hpp"
#include "Core/Tensor.hpp"

#include "Functional/Functional.hpp"
#include "Functional/FunctionnalRG.hpp"
#include "Physics/DiffGeometry/Metric.hpp" // Clairement identifié comme Physique
