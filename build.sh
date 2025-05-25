#!/usr/bin/env bash
set -euo pipefail

#---------------------------------------------------------------------#
# A quick build script for Morpheus_lib with pybind11 and Python 3.12 #
#---------------------------------------------------------------------#

# Allow overriding the Python executable via environment variable
PYTHON_EXE=${PYTHON_EXE:-$(which python3.12)}

# Get the pybind11 CMake directory for the chosen Python
PYBIND11_DIR=$($PYTHON_EXE -m pybind11 --cmakedir)

# Create and enter the build directory
echo "Creating build directory 'pybuild' and configuring project..."
mkdir -p pybuild
tcd=pybuild
cd "$tcd"

echo "Using Python executable: $PYTHON_EXE"
echo "Using pybind11 CMake dir: $PYBIND11_DIR"

echo "Running CMake..."
cmake \
  -DPYTHON_EXECUTABLE="$PYTHON_EXE" \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  ..

echo "Building with make..."
# Use all available cores for parallel build
total_cores=$(nproc || echo 1)
make -j"$total_cores"

echo "Build complete!"
