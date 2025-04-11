NAME        = Morpheus
BENCH_NAME  = benchmark

CC          = clang++

SRC_DIR     = srcs
BENCH_DIR   = $(SRC_DIR)/benchmarks
INC_DIR     = includes
OBJ_DIR     = build

CFLAGS      = -O3 -std=c++17 -mfma -mavx2 -mtune=native -fopenmp -I$(INC_DIR)
BLAS_FLAGS  = -lblas -lm -lopenblas

ifeq ($(VERBOSE), true)
    CFLAGS += -D VERBOSE
endif

ifeq ($(DEBUG), true)
    CFLAGS += -g
endif

SRC     = $(shell find $(SRC_DIR) -name '*.cpp' ! -path "$(BENCH_DIR)/*")
OBJ     = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC))

BENCH_SRC = $(BENCH_DIR)/bench.cpp
BENCH_OBJ = $(OBJ_DIR)/bench.o

.PHONY: all clean fclean re benchmark

all: $(NAME)
	@echo -e "\033[1;32mBuild complete!\033[0m"

$(NAME): $(OBJ)
	@echo -e "\033[1;33mLinking $@...\033[0m"
	$(CC) $(CFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

benchmark: $(BENCH_OBJ)
	@echo -e "\033[1;36mCompiling benchmark...\033[0m"
	$(CC) $(CFLAGS) -o $(BENCH_NAME) $^ $(BLAS_FLAGS)

$(OBJ_DIR)/bench.o: $(BENCH_SRC) | $(OBJ_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	@echo -e "\033[1;31mCleaning up object files...\033[0m"
	rm -rf $(OBJ_DIR)

fclean: clean
	rm -f $(NAME) $(BENCH_NAME)

re: fclean all
