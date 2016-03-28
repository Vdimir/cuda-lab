
CC=nvcc

task3: src/task3.cu
	$(CC) -O3 src/task3.cu -lopencv_core -lopencv_highgui -o task3
	
run: task3
	optirun ./task3