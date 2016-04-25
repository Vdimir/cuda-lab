
CC=nvcc
OUT_DIR=build
CFLAGS=

build: $(OUT_DIR)/task3

$(OUT_DIR)/task3: src/task3.cu
	$(CC) -O3 -lopencv_core -lopencv_highgui -o $(OUT_DIR)/task3 $(CFLAGS) src/task3.cu

dir:
	mkdir -p $(OUT_DIR)

run: $(OUT_DIR)/task3
	$(OUT_DIR)/task3
