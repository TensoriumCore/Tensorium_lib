NAME         = Morpheus
BENCH_NAME   = benchmark
LIB_NAME     = libmorpheus.so

CC           = clang++
SRC_DIR      = tests
BENCH_DIR    = $(SRC_DIR)/benchmarks
PLGIN_DIR    = Plugins
INC_DIR      = includes
OBJ_DIR      = build

PLUGIN_SRC   = $(PLGIN_DIR)/MorpheusDispatchPlugin.cpp
PLUGIN_OUT   = $(PLGIN_DIR)/MorpheusDispatchPlugin.so
LLVM_IR_PLUGIN_SRC  = $(PLGIN_DIR)/MorpheusLLVM_IRCheck.cpp
LLVM_IR_PLUGIN_OUT  = $(PLGIN_DIR)/MorpheusLLVM_IRCheck.so
LLVM_IR_LIBS        := $(shell llvm-config --ldflags --system-libs --libs core passes)

LLVM_CXXFLAGS:= $(shell llvm-config --cxxflags)
LLVM_LDFLAGS := $(shell llvm-config --ldflags --system-libs --libs all)
CLANG_LIBS   := -lclangFrontend -lclangTooling -lclangBasic -lclangLex

CXX_STD      = -std=c++17
BASE_FLAGS   = -O3 -fopenmp -mtune=native -g -I$(INC_DIR) -Wno-ignored-attributes -Wignored-attributes -Rpass-analysis=morpheus-align

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

PLUGIN_FLAGS = -Xclang -load -Xclang $(PLUGIN_OUT) \
               -Xclang -add-plugin -Xclang morpheus-dispatch

LLVM_IR_PLUGIN_FLAGS = -fpass-plugin=$(LLVM_IR_PLUGIN_OUT) \



SRC  := $(shell find $(SRC_DIR) -name '*.cpp' ! -path "$(BENCH_DIR)/*")
OBJ  := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC))

LIB_SRC := $(filter-out $(SRC_DIR)/main.cpp,$(SRC))
LIB_OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(LIB_SRC))

BENCH_SRC = $(BENCH_DIR)/bench.cpp
BENCH_OBJ = $(OBJ_DIR)/bench.o

BLAS_FLAGS = -lblas -lm -lopenblas

.PHONY: all clean fclean re benchmark lib help plugin plugin-test llvm-ir-plugin

all: plugin $(NAME)

$(NAME): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR) plugin llvm-ir-plugin
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(PLUGIN_FLAGS) $(LLVM_IR_PLUGIN_FLAGS) -fPIC -c $< -o $@


$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

benchmark: $(BENCH_OBJ)
	$(CC) $(CFLAGS) $(PLUGIN_FLAGS) -o $(BENCH_NAME) $^ $(BLAS_FLAGS) $(LDFLAGS)

$(OBJ_DIR)/bench.o: $(BENCH_SRC) | $(OBJ_DIR) plugin
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(PLUGIN_FLAGS) -fPIC -c $< -o $@

lib: $(LIB_NAME)

$(LIB_NAME): $(LIB_OBJ)
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LDFLAGS)

llvm-ir-plugin: $(LLVM_IR_PLUGIN_OUT)

$(LLVM_IR_PLUGIN_OUT): $(LLVM_IR_PLUGIN_SRC)
	$(CC) -fPIC -shared -o $@ $< $(LLVM_IR_FLAGS) $(LLVM_IR_LIBS) -std=c++17

plugin: $(PLUGIN_OUT)

$(PLUGIN_OUT): $(PLUGIN_SRC)
	$(CC) -fPIC -shared -o $@ $< $(LLVM_CXXFLAGS) $(LLVM_LDFLAGS) $(CLANG_LIBS)

plugin-test: plugin
	$(CC) -S -emit-llvm $(PLUGIN_FLAGS) Plugins/test.cpp -o Plugins/a.out

clean:
	rm -rf $(OBJ_DIR)

fclean: clean
	rm -f $(NAME) $(BENCH_NAME) $(LIB_NAME) $(PLUGIN_OUT) $(PLGIN_DIR)/a.out

help:
	@echo "Targets:  all benchmark lib plugin plugin-test clean fclean re"

re: fclean all
