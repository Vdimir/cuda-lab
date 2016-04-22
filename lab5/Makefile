
CC=nvcc
OUT_DIR=build
CFLAGS=

build: $(OUT_DIR)/task5

$(OUT_DIR)/task5: main.cpp
	$(CC) -O3 -lOpenCL -lopencv_core -lopencv_highgui -o $(OUT_DIR)/task5 $(CFLAGS) main.cpp
	
dir:
	mkdir -p $(OUT_DIR)
    
run: $(OUT_DIR)/task5
	optirun $(OUT_DIR)/task5