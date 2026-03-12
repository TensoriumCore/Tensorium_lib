# Tensorium MPI Module

Distributed-memory parallelization for BSSN numerical relativity simulations.

## Architecture

```
Tensorium/MPI/
├── MPI.hpp                 # Master include
├── MPIContext.hpp          # RAII MPI init/finalize
├── MPIDomain.hpp           # 3D Cartesian domain decomposition
├── MPIHaloExchange.hpp     # Non-blocking halo exchange with MPI datatypes
├── MPIReductions.hpp       # Global reductions (CFL, norms)
├── MPIBSSNIntegration.hpp  # BSSN grid factory and boundary adapter
└── MPIBSSNRK4.hpp          # MPI-aware RK4 time stepper
```

## Quick Start

```cpp
#define TENSORIUM_ENABLE_MPI
#include "Tensorium/MPI/MPI.hpp"
#include "Tensorium/MPI/MPIBSSNRK4.hpp"

int main(int argc, char** argv) {
    using namespace tensorium::mpi;

    // Initialize MPI with thread support for OpenMP
    MPIContext ctx(argc, argv, ThreadLevel::Funneled);

    // Configure domain (128³ grid, 256M box, 6 ghost cells)
    auto config = make_domain_config(128, 128, 128, 6, 256.0);

    // Create Cartesian decomposition
    MPIDomain domain(config);
    domain.print_info();

    // Create local BSSN grid for this process
    auto grid = create_local_bssn_grid<double>(domain);

    // Create MPI stepper
    MPIBSSNRKStepper<double> stepper(domain, *grid);

    // Evolution loop
    for (size_t step = 0; step < nsteps; ++step) {
        double dt = stepper.compute_dt(*grid, 0.25);
        stepper.step(*grid, dt, step);
    }

    return 0;
}
```

## Compilation

```bash
mkdir build && cd build
cmake .. -DUSE_MPI=ON
make TensoriumMPIMovingPuncture
```

## Running on Xeon Phi (KNL) with Mellanox InfiniBand

### Intel MPI (Recommended for KNL)

```bash
# Load Intel environment
source /opt/intel/oneapi/setvars.sh

# Compile with KNL optimizations
cmake .. -DUSE_MPI=ON -DUSE_KNL=ON -DCMAKE_CXX_FLAGS="-xMIC-AVX512"

# Create hostfile
cat > hostfile << EOF
node1 slots=1
node2 slots=1
EOF

# Run with optimal settings for KNL + Mellanox
mpirun -n 2 -ppn 1 -hostfile hostfile \
    -genv OMP_NUM_THREADS=64 \
    -genv OMP_PROC_BIND=spread \
    -genv OMP_PLACES=threads \
    -genv KMP_AFFINITY=granularity=fine,balanced \
    -genv I_MPI_FABRICS=shm:ofi \
    -genv FI_PROVIDER=verbs \
    numactl --preferred=1 ./TensoriumMPIMovingPuncture \
        --nx 128 --steps 1000 --proc-z 2
```

### OpenMPI with UCX

```bash
# Configure UCX for Mellanox ConnectX-3
export UCX_NET_DEVICES=mlx4_0:1,mlx4_1:1  # Both ports
export UCX_TLS=rc,sm,self                  # RDMA + shared memory

mpirun -np 2 --hostfile hostfile \
    --map-by ppr:1:node:PE=64 \
    -x OMP_NUM_THREADS=64 \
    -x OMP_PROC_BIND=spread \
    -x UCX_NET_DEVICES \
    -x UCX_TLS \
    ./TensoriumMPIMovingPuncture --nx 128 --steps 1000
```

## KNL Memory Configuration (MCDRAM)

The Intel Xeon Phi 7200 has 16GB of high-bandwidth MCDRAM. For best performance:

```bash
# Flat mode (MCDRAM as NUMA node 1)
numactl --preferred=1 ./TensoriumMPIMovingPuncture ...

# Or use memkind in code (requires -DUSE_KNL=ON)
export MEMKIND_HBW_NODES=1
```

## Domain Decomposition

For 2 processes, the default is 1×1×2 decomposition (split in Z):

```
Process 0: z ∈ [z_min, z_mid]
Process 1: z ∈ [z_mid, z_max]
```

This minimizes communication surface area. You can override:

```bash
./TensoriumMPIMovingPuncture --proc-z 2  # Explicit Z-split
```

For more processes:
- 4 procs: 1×2×2 or 2×2×1
- 8 procs: 2×2×2
- MPI_Dims_create auto-selects if all are 0

## Halo Exchange

Each RK4 sub-step exchanges ghost zones:

1. **Pack**: Interior cells at subdomain boundaries
2. **Exchange**: MPI_Isend/Irecv with MPI derived datatypes (no manual packing)
3. **Unpack**: Received data fills ghost zones
4. **Physical BC**: Applied at true domain boundaries only

## Performance Tips

1. **Overlap compute/comm**: The halo exchanger supports non-blocking operations
2. **NUMA-aware allocation**: Use `numactl --preferred=1` for MCDRAM on KNL
3. **Thread pinning**: Use `KMP_AFFINITY` or `OMP_PLACES` for cache efficiency
4. **Network tuning**:
   - For ConnectX-3: `mlx4_0` device
   - For ConnectX-4+: `mlx5_0` device
   - Enable RDMA with `UCX_TLS=rc`

## Troubleshooting

### MPI_Init fails with fabric error
```bash
export I_MPI_FABRICS=shm:tcp  # Fall back to TCP if InfiniBand fails
```

### Poor scaling
- Check `numactl -H` to verify NUMA topology
- Ensure ghost zone width (ng=6) is sufficient for stencil order
- Monitor with `mpirun -genv I_MPI_DEBUG=5`

### Constraint violations grow
- Reduce CFL factor: `--cfl 0.1`
- Increase resolution: `--nx 192`
- Check physical boundary conditions at domain edges

## API Reference

### MPIContext
RAII wrapper for `MPI_Init_thread` / `MPI_Finalize`.

### MPIDomain
3D Cartesian topology with automatic neighbor detection.

### HaloExchanger<T>
MPI datatype-based halo exchange (no manual buffer packing).

### Reductions
Global `Allreduce` operations for CFL, norms, etc.

### MPIBSSNRKStepper<T>
Full MPI-parallel RK4 time integrator with constraint monitoring.
