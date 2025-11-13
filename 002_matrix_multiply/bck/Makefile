NVCC = nvcc
# Auto-detect GPU architecture, or use sm_75 as default
# Convert "8.6" to "sm_86", "7.5" to "sm_75", etc.
GPU_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | awk -F. '{printf "sm_%d%d", $$1, $$2}' || echo "sm_75")
NVCC_FLAGS = -O2 -arch=$(GPU_ARCH)
TARGET = a.out
SOURCES = main.cu kernel_naive.cu kernel_coalescing.cu
OBJECTS = $(SOURCES:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.cu kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

# Print detected GPU architecture
info:
	@echo "Detected GPU architecture: $(GPU_ARCH)"
	@echo "Compile with: make"
	@echo "Run with: ./$(TARGET)"

.PHONY: all clean info

