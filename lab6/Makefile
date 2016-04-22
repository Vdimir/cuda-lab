
CC=nvcc
OUT_DIR=build
EXE_FILE_NAME=lab6

SRC_DIR=src
SRC_FILE_NAME=lab6.cu 
CFLAGS=

build: $(OUT_DIR)/$(EXE_FILE_NAME)

$(OUT_DIR)/$(EXE_FILE_NAME): $(SRC_DIR)/$(SRC_FILE_NAME)
	$(CC) -O3 -lopencv_core -lopencv_highgui -o $(OUT_DIR)/$(EXE_FILE_NAME) $(CFLAGS) $(SRC_DIR)/$(SRC_FILE_NAME)
		
dir:
	mkdir -p $(OUT_DIR)
    
run: $(OUT_DIR)/$(EXE_FILE_NAME)
	optirun $(OUT_DIR)/$(EXE_FILE_NAME)