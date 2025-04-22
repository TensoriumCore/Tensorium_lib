NAME         = Morpheus
BENCH_NAME   = benchmark
LIB_NAME     = libmorpheus.so

CC           = clang++
SRC_DIR      = tests
BENCH_DIR    = $(SRC_DIR)/benchmarks
INC_DIR      = includes
OBJ_DIR      = build

CXX_STD      = -std=c++17
BASE_FLAGS   = -O3 -fopenmp -mtune=native -g -I$(INC_DIR)

AVX2_FLAGS   = -mfma -mavx2
AVX512_FLAGS = -mfma -mavx512f 

CFLAGS       = $(CXX_STD) $(BASE_FLAGS) $(AVX2_FLAGS)

ifeq ($(AVX512), true)
	CFLAGS := $(CXX_STD) $(BASE_FLAGS) $(AVX512_FLAGS)
endif

ifeq ($(VERBOSE), true)
	CFLAGS += -DVERBOSE
endif

ifeq ($(DEBUG), true)
	CFLAGS += -g
endif

ifeq ($(USE_MPI), true)
	CFLAGS += -DMORPHEUS_USE_MPI
	LDFLAGS += -lmpi
endif

ifeq ($(USE_KNL), true)
	CFLAGS += -DUSE_KNL -mtune=knl -mfma -mavx512f -mavx512cd
	LDFLAGS += -lmemkind 
endif

SRC         = $(shell find $(SRC_DIR) -name '*.cpp' ! -path "$(BENCH_DIR)/*")
OBJ         = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC))

LIB_SRC     = $(filter-out $(SRC_DIR)/main.cpp, $(SRC))
LIB_OBJ     = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(LIB_SRC))

BENCH_SRC   = $(BENCH_DIR)/bench.cpp
BENCH_OBJ   = $(OBJ_DIR)/bench.o

BLAS_FLAGS  = -lblas -lm -lopenblas

.PHONY: all clean fclean re benchmark lib help

all: $(NAME)

$(NAME): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

benchmark: $(BENCH_OBJ)
	$(CC) $(CFLAGS) -o $(BENCH_NAME) $^ $(BLAS_FLAGS) $(LDFLAGS)

$(OBJ_DIR)/bench.o: $(BENCH_SRC) | $(OBJ_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

lib: $(LIB_NAME)

$(LIB_NAME): $(LIB_OBJ)
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(OBJ_DIR)

fclean: clean
	rm -f $(NAME) $(BENCH_NAME) $(LIB_NAME)

help:
	@echo "Makefile for Morpheus"
	@echo "Usage:"
	@echo "  make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build the executable"
	@echo "  benchmark  - Build the benchmark executable"
	@echo "  lib        - Build the shared library"
	@echo "  clean      - Remove object files and directories"
	@echo "  fclean     - Remove all generated files (executables, libraries, object files)"
	@echo "  re         - Rebuild everything"
	@echo ""
	@echo "Options:"
	@echo "  AVX512=true  - Enable AVX512 optimizations"
	@echo "  VERBOSE=true - Enable verbose output"
	@echo "  DEBUG=true   - Enable debug symbols"
	@echo "  USE_MPI=true - Enable MPI support"
	@echo "  USE_KNL=true - Enable KNL support"

re: fclean all
