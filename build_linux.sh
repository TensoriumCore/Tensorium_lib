#!/usr/bin/env bash
set -e

mkdir -p pybuild
cd pybuild

cmake .. \
    -DPYTHON_EXECUTABLE=$(which python) \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=.

make -j$(nproc)

SO_FILE=$(find . -maxdepth 1 -name "*.so" | head -n 1)

if [ -z "$SO_FILE" ]; then
    echo "Erreur : aucun .so généré"
    exit 1
fi

mv "$SO_FILE" ../pysrc/tensorium/
