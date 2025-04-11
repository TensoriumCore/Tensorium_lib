#!/usr/bin/env zsh

# Colors
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
NC=$(tput sgr0)

# Build
mkdir -p pybuild
cd pybuild
cmake .. && make -j4
cd ..

echo ""
echo "${GREEN}Build complete!${NC}"
echo "${GREEN}Running tests...${NC}"
echo ""

# Run Python tests
python3 test_python_api.py

echo ""
echo "${GREEN}Running C++ benchmarks...${NC}"
echo ""

make benchmark -j
./benchmark

echo ""
echo "${GREEN}Running tests...${NC}"
echo ""
make -j
./Morpheus
