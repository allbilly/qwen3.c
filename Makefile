# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc
.DEFAULT_GOAL := run

NPU_SRCS = npu_matmul.c
NPU_INCLUDES = -I../include
NPU_LIBS = -ldrm

# Native Asahi Linux M1 backend. Uses the ~/ane KMD ABI without libdrm or RKNN.
ANE_FLAGS ?= -O3 -fopenmp -march=native
ANE_SRCS = ane/ane_matmul.c
ANE_HEADERS = ane/ane_matmul.h ane/linear_template.h npu_matmul.h qwen3.h
.PHONY: ane cpu ane-test
ane: runq-ane
runq-ane: runq.c $(ANE_SRCS) $(ANE_HEADERS)
	$(CC) $(ANE_FLAGS) -D_FILE_OFFSET_BITS=64 -DQWEN3_USE_ANE -I. runq.c $(ANE_SRCS) -lm -o $@
cpu: runq-cpu
runq-cpu: runq.c npu_matmul.c npu_matmul.h qwen3.h
	$(CC) $(ANE_FLAGS) -D_FILE_OFFSET_BITS=64 -DQWEN3_DISABLE_NPU runq.c npu_matmul.c -lm -o $@
ane-test: tests/ane_matmul_test
	python3 tools/benchmark_queue.py -- ./tests/ane_matmul_test
tests/ane_matmul_test: tests/ane_matmul_test.c $(ANE_SRCS) $(ANE_HEADERS)
	$(CC) $(ANE_FLAGS) -D_FILE_OFFSET_BITS=64 -DQWEN3_USE_ANE -I. $< $(ANE_SRCS) -lm -o $@

# Direct-register FP16 engine for RK3588 and the matched Qwen3-0.6B checkpoint.
FP16_NPU_INCLUDE ?= ../npu/include
FP16_SRCS = fp16/run.c fp16/fp16_backend.c
FP16_HEADERS = fp16/fp16_backend.h fp16/prefill.h fp16/attention_backend.h fp16/vector_math.h qwen3.h npu_matmul.h
.PHONY: fp16
fp16: runq-fp16
runq-fp16: $(FP16_SRCS) $(FP16_HEADERS) runq.c
	$(CC) -Ofast -fopenmp -march=native -D_FILE_OFFSET_BITS=64 -I$(FP16_NPU_INCLUDE) $(FP16_SRCS) -lm -ldrm -o $@

# Isolated phase-routing experiment. The selected fp16 target stays separate.
FP16_ROUTE_DIR = fp16/experiments/yalm
FP16_ROUTE_SRCS = $(FP16_ROUTE_DIR)/source/fp16/run.c $(FP16_ROUTE_DIR)/npu_backend.c $(FP16_ROUTE_DIR)/source/fp16/gpu_attention.c $(FP16_ROUTE_DIR)/router.c $(FP16_ROUTE_DIR)/linear_gpu.c
FP16_ROUTE_HEADERS = $(wildcard $(FP16_ROUTE_DIR)/*.h $(FP16_ROUTE_DIR)/source/*.h $(FP16_ROUTE_DIR)/source/fp16/*.h)
.PHONY: fp16-routes
fp16-routes: runq-fp16-routes
$(FP16_ROUTE_DIR)/linear_source.h: $(FP16_ROUTE_DIR)/linear.cl $(FP16_ROUTE_DIR)/embed.py
	python3 $(FP16_ROUTE_DIR)/embed.py $< $@ linear_source
$(FP16_ROUTE_DIR)/source/fp16/gpu_source.h: $(FP16_ROUTE_DIR)/attention.cl $(FP16_ROUTE_DIR)/embed.py
	python3 $(FP16_ROUTE_DIR)/embed.py $< $@ gpu_source
runq-fp16-routes: $(FP16_ROUTE_SRCS) $(FP16_ROUTE_HEADERS) $(FP16_ROUTE_DIR)/source/runq.c
	$(CC) -Ofast -fopenmp -march=native -D_FILE_OFFSET_BITS=64 -I$(FP16_NPU_INCLUDE) $(FP16_ROUTE_SRCS) -lm -ldrm -ldl -l:libOpenCL.so.1 -o $@

# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: runq.c
	$(CC) -O3 -o runq -D_FILE_OFFSET_BITS=64 runq.c $(NPU_SRCS) $(NPU_INCLUDES) -lm $(NPU_LIBS)

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
debug: runq.c
	$(CC) -g -o runq -D_FILE_OFFSET_BITS=64 runq.c $(NPU_SRCS) $(NPU_INCLUDES) -lm $(NPU_LIBS)

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: fast
fast: runq.c
	$(CC) -Ofast -o runq -D_FILE_OFFSET_BITS=64 runq.c $(NPU_SRCS) $(NPU_INCLUDES) -lm $(NPU_LIBS)

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./runq out/model.bin
.PHONY: openmp
openmp: runq.c
	$(CC) -Ofast -fopenmp -march=native -D_FILE_OFFSET_BITS=64 runq.c $(NPU_SRCS) $(NPU_INCLUDES) -lm $(NPU_LIBS) -o runq

.PHONY: win64
win64:
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -D_FILE_OFFSET_BITS=64 -DQWEN3_DISABLE_NPU -o runq.exe -I. runq.c $(NPU_SRCS) win.c

# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: gnu
gnu:
	$(CC) -Ofast -std=gnu11 -o runq -D_FILE_OFFSET_BITS=64 runq.c $(NPU_SRCS) $(NPU_INCLUDES) -lm $(NPU_LIBS)

.PHONY: gnuopenmp
gnuopenmp:
	$(CC) -Ofast -fopenmp -std=gnu11 -D_FILE_OFFSET_BITS=64 runq.c $(NPU_SRCS) $(NPU_INCLUDES) -lm $(NPU_LIBS) -o runq

.PHONY: clean
clean:
	rm -f runq
	rm -f runq-fp16
	rm -f runq-fp16-routes
	rm -f runq-ane runq-cpu runq-ane-bench tests/ane_matmul_test
