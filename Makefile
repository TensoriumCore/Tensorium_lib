###############################################################################
# Tensorium – unified Makefile (macOS / Linux)                                #
###############################################################################
#  Edit only the high‑level toggles below; the rest is automatic.
###############################################################################

# ─────────────────────────────── Build toggles ───────────────────────────────
AVX512  ?= false   # true → compile with AVX‑512
VERBOSE ?= false   # true → extra runtime logs
DEBUG   ?= false   # true → keep symbols (-g)
USE_MPI ?= false   # true → link MPI
USE_KNL ?= false   # true → tune for Intel KNL (+memkind)


# ─────────────────────────────── Path settings ───────────────────────────────
NAME        := Tensorium                 # main executable
LIB_NAME    := libtensorium.so           # shared library
SRC_DIR     := tests
BENCH_DIR   := $(SRC_DIR)/benchmarks
PLUGIN_DIR  := Plugins
INC_DIR     := includes
OBJ_DIR     := build

# ─────────────────────────────── Source discovery ────────────────────────────
SRC        := $(shell find $(SRC_DIR) -name '*.cpp' ! -path '$(BENCH_DIR)/*')
OBJ        := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC))
LIB_SRC    := $(filter-out $(SRC_DIR)/main.cpp,$(SRC))
LIB_OBJ    := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(LIB_SRC))
BENCH_SRC  := $(BENCH_DIR)/bench.cpp
BENCH_OBJ  := $(OBJ_DIR)/bench.o
PLUGIN_SRC := $(PLUGIN_DIR)/TensoriumDispatchPlugin.cpp
LLVM_IR_SRC:= $(PLUGIN_DIR)/TensoriumLLVM_IRCheck.cpp

# ───────────────────────────────── LLVM / Clang ──────────────────────────────
CC            := clang++
CXX_STD       := -std=c++17 -g
LLVM_CXXFLAGS := $(shell llvm-config --cxxflags)
LLVM_LDFLAGS  := $(shell llvm-config --ldflags --system-libs)
LLVM_IR_LIBS  := $(shell llvm-config --ldflags --system-libs --libs core passes)

# — filter out bogus -L towards empty clang/lib directories (macOS Nix) —
LLVM_LDFLAGS  := $(filter-out -L/nix/store/%-clang-*/lib,$(LLVM_LDFLAGS))

# ─────────────────────────────── Compiler flags ──────────────────────────────
BASE_FLAGS := -O3 -fopenmp -mtune=native -I$(INC_DIR) \
              -Wno-ignored-attributes -Wignored-attributes \
              -Rpass-analysis=tensorium-align
AVX2_FLAGS   := -mfma -mavx2
AVX512_FLAGS := -mfma -mavx512f -mavx512cd

# — drop -mtune=native if Nix forbids impure flags —
ifeq ($(NIX_ENFORCE_NO_NATIVE),1)
  BASE_FLAGS := $(filter-out -mtune=native,$(BASE_FLAGS))
endif

CFLAGS := $(CXX_STD) $(BASE_FLAGS) $(AVX2_FLAGS)
ifeq ($(AVX512),true)
  CFLAGS := $(CXX_STD) $(BASE_FLAGS) $(AVX512_FLAGS)
endif
ifeq ($(VERBOSE),true)
  CFLAGS += -DVERBOSE
endif
ifeq ($(DEBUG),true)
  CFLAGS += -g
endif
ifeq ($(USE_MPI),true)
  CFLAGS += -DMORPHEUS_USE_MPI
  EXTRA_LDFLAGS += -lmpi
endif
ifeq ($(USE_KNL),true)
  CFLAGS += -DUSE_KNL -mtune=knl -mfma -mavx512f -mavx512cd
  EXTRA_LDFLAGS += -lmemkind
endif

# ───────────────────────────── Platform‑specific plugin ──────────────────────
OS := $(shell uname -s)
ifeq ($(OS),Darwin)                    # — macOS —
  PLUGIN_EXT     := dylib
  PLUGIN_LIBS    := -lclang-cpp
  PLUGIN_LDFLAGS := -Wl,-undefined,dynamic_lookup
else                                   # — Linux / other —
  PLUGIN_EXT     := so
  PLUGIN_LIBS    := -lclangTooling -lclangFrontend -lclangDriver \
                    -lclangSerialization -lclangCodeGen -lclangParse \
                    -lclangSema -lclangEdit -lclangAnalysis \
                    -lclangASTMatchers -lclangAST -lclangLex -lclangBasic -lLLVM
  PLUGIN_LDFLAGS :=
endif
PLUGIN_OUT  := $(PLUGIN_DIR)/TensoriumDispatchPlugin.$(PLUGIN_EXT)
LLVM_IR_OUT := $(PLUGIN_DIR)/TensoriumLLVM_IRCheck.so

# — helper flags injected during compilation —
PLUGIN_FLAGS         := -Xclang -load -Xclang $(PLUGIN_OUT) -Xclang -add-plugin -Xclang tensorium-dispatch
LLVM_IR_PLUGIN_FLAGS := -fpass-plugin=$(LLVM_IR_OUT)
BLAS_FLAGS           := -lblas -lm -lopenblas
LDFLAGS              := $(LLVM_LDFLAGS) $(EXTRA_LDFLAGS)

# ─────────────────────────────── Phony targets ───────────────────────────────
.PHONY: all lib benchmark plugin llvm-ir-plugin plugin-test clean fclean re help

# ─────────────────────────────── Build graph ─────────────────────────────────
all: plugin $(NAME)

$(NAME): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR) plugin llvm-ir-plugin
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(PLUGIN_FLAGS) $(LLVM_IR_PLUGIN_FLAGS) -fPIC -c $< -o $@

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

lib: $(LIB_NAME)
$(LIB_NAME): $(LIB_OBJ)
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LDFLAGS)

benchmark: $(BENCH_OBJ)
	$(CC) $(CFLAGS) $(PLUGIN_FLAGS) -o benchmark $^ $(BLAS_FLAGS) $(LDFLAGS)

$(OBJ_DIR)/bench.o: $(BENCH_SRC) | $(OBJ_DIR) plugin
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(PLUGIN_FLAGS) -fPIC -c $< -o $@

# — Clang plugin (AST) —
plugin: $(PLUGIN_OUT)
$(PLUGIN_OUT): $(PLUGIN_SRC)
	$(CC) -fPIC -shared -o $@ $< $(LLVM_CXXFLAGS) $(LDFLAGS) $(PLUGIN_LIBS) $(PLUGIN_LDFLAGS)

# — LLVM‑IR pass plugin —
llvm-ir-plugin: $(LLVM_IR_OUT)
$(LLVM_IR_OUT): $(LLVM_IR_SRC)
	$(CC) -fPIC -shared -o $@ $< $(LLVM_CXXFLAGS) $(LLVM_IR_LIBS) -std=c++17

plugin-test: plugin
	$(CC) -S -emit-llvm $(PLUGIN_FLAGS) $(PLUGIN_DIR)/test.cpp -o $(PLUGIN_DIR)/a.out

clean:
	rm -rf $(OBJ_DIR)

fclean: clean
	rm -f $(NAME) benchmark $(LIB_NAME) $(PLUGIN_OUT) $(PLUGIN_DIR)/a.out $(LLVM_IR_OUT)

re: fclean all

help:
	@echo "Targets:  all  lib  benchmark  plugin  llvm-ir-plugin  plugin-test  clean  fclean  re"
