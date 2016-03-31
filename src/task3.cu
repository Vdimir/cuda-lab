#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#define CUDA_TIMER_START(id) cudaEvent_t start##id, stop##id; \
	    cudaEventCreate(&start##id); \
	    cudaEventCreate(&stop##id); \
	    cudaEventRecord(start##id, 0)

#define CUDA_TIMER_STOP(id, elapsedTime) float elapsedTime = -1;\
		cudaEventRecord(stop##id, 0); \
	    cudaEventSynchronize(stop##id); \
	    cudaEventElapsedTime(&elapsedTime, start##id, stop##id); \
	    cudaEventDestroy(start##id); \
	    cudaEventDestroy(stop##id); \
	    elapsedTime /= 1000.0;

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        std::cerr<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }


#define COMP_CNT 3

float rotate_array_cpu(uchar *source, uchar *target, int m, int n) {

	float elapsedTimeCPU = -1;
    clock_t startCPU = clock();

	for (int i = 0; i < m; ++i) {
		for (int j = n-1; j >=0; j--) {
	      	int SIdx = (i*n + (n-1 - j));
	       	int DIdx = (j*m + i);

	       	((uchar3*)target)[DIdx] = ((uchar3*)source)[SIdx];
		}
	}

    elapsedTimeCPU = (float)(clock()-startCPU)/CLOCKS_PER_SEC;

    return elapsedTimeCPU;
}

__global__ void rotate_gpu_kernel(const uchar *src, uchar *trg, int m, int n) {
	int tx = threadIdx.x,
			ty = threadIdx.y;
	int bx = blockIdx.x*blockDim.x,
			by = blockIdx.y*blockDim.y;
    int i = tx + bx;
    int j = ty + by;

    if (i >= m || j >= n) return;
    int step = 3;
 	int SIdx = (i*n + (n-1 - j)) * step;
 	int DIdx = (j*m + i) * step;

   	//((uchar3*)trg)[DIdx] = ((uchar3*)src)[SIdx];
   	for (int c = 0; c < COMP_CNT; c++)
   		trg[DIdx+c] = src[SIdx+c];
}


#define BLOCK_H 16
#define BLOCK_W 16

#define TILE_H 16
#define TILE_W 16

__global__ void rotate_gpu_kernel_shared(uchar *src, uchar *trg, int m, int n) {
	int tx = threadIdx.x,
			ty = threadIdx.y;

	int bx = blockIdx.x*blockDim.x,
			by = blockIdx.y*blockDim.y;

    __shared__ uchar3 block[TILE_H][TILE_W+4];

	int DIdx = (n-1-(tx + by) + (ty + bx)*n);

	if (n-1-(tx + by) >= 0 && (ty + bx) < m) {
		block[tx][ty] = ((uchar3*)src)[DIdx];
	}

    __syncthreads();


	int SIdx = ((tx + bx) + (ty + by)*m);

	if ((tx + bx) < m && (ty + by) < n) {
		((uchar3*)trg)[SIdx] = block[ty][tx];
	}

}


void call_kernel_shared(uchar *dev_src, uchar *dev_res, int m, int n) {
	const int bsx = BLOCK_W, bsy = BLOCK_H;
    dim3 threadsPerBlock(bsx, bsy);
    dim3 numBlocks((m + (bsx-1)) / bsx, (n + (bsy-1)) / bsy);

    rotate_gpu_kernel_shared<<<numBlocks, threadsPerBlock >>>(dev_src, dev_res, m, n);
}

void call_kernel_simple(uchar *dev_src, uchar *dev_res, int m, int n) {
	const int bsx = 16, bsy = 16;
    dim3 threadsPerBlock(bsx, bsy);
    dim3 numBlocks((m + (bsx-1)) / bsx, (n + (bsy-1)) / bsy);

    rotate_gpu_kernel<<<numBlocks, threadsPerBlock >>>(dev_src, dev_res, m, n);
}

template<typename F>
float rotate_array_gpu(F kernel_call, uchar *h_src, uchar *h_res,  int m, int n) {

	int N = m*n*COMP_CNT;
    uchar *dev_src;
    uchar *dev_res;


    cudaMalloc(&dev_src, N);
    cudaMalloc(&dev_res, N);

    CHECK(cudaMemcpy(dev_src, h_src, N, cudaMemcpyHostToDevice));

    CUDA_TIMER_START(A);

    kernel_call(dev_src, dev_res, m, n);

    CUDA_TIMER_STOP(A, elapsedTime);

    CHECK(cudaMemcpy(h_res, dev_res, N, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dev_src));
    CHECK(cudaFree(dev_res));
    return elapsedTime;
}


enum Computing { CPU, GPU_SIMPLE, GPU_SHARED };

void rotate_array_common(Computing runOn,  uchar *h_src, uchar *h_res,  int m, int n) {
	float elapsedTime = -1;
	switch (runOn) {
		case CPU:
			elapsedTime = rotate_array_cpu(h_src, h_res, m, n);
		    std::cout << "CPU time: "<< elapsedTime << std::endl;
			break;

		case GPU_SIMPLE:
			elapsedTime = rotate_array_gpu(call_kernel_simple, h_src, h_res, m, n);
		    std::cout << "CUDA Simp time: "<< elapsedTime << std::endl;
			break;

		case GPU_SHARED:
			elapsedTime = rotate_array_gpu(call_kernel_shared, h_src, h_res, m, n);
		    std::cout << "CUDA Shar time: "<< elapsedTime << std::endl;
			break;

		default:
			break;
	}
	return;
}

void rotate_picture_test(Computing runOn, const char * in_file_name, const char * out_file_name) {
	cv::Mat image = cv::imread(in_file_name, CV_LOAD_IMAGE_COLOR);

	if (!image.data) {
		std::cout << "Error open image" << std::endl;
		exit(1);
	}
	int n = image.cols,
			m = image.rows;

	cv::Mat rotated(n, m, CV_8UC3);

	rotate_array_common(runOn, image.data, rotated.data, m, n);

    cv::imwrite(out_file_name, rotated);
    return;
}

bool diff_img(const char * im_first, const char * im_second) {
	cv::Mat ima = cv::imread(im_first, CV_LOAD_IMAGE_COLOR);
	cv::Mat imb = cv::imread(im_second, CV_LOAD_IMAGE_COLOR);

	int m, n;
	if (((m = ima.rows) != imb.rows) ||
		((n = ima.cols) != imb.cols)) {
		return false;
	}

	for (int i = 0; i < m*n*3; ++i) {
		if (ima.data[i] != imb.data[i]) {
			return false;
		}
	}
    return true;
}

#define ASSERT(cond, msg) if (!cond) {\
	std::cerr << msg << std::endl;\
	exit(1);\
}

int main(int argc, char** argv) {
	char *fname;
	if (argc <= 1) {
		fname = (char*)"pic.jpg";
	} else {
		fname = argv[1];
	}

	rotate_picture_test(GPU_SHARED, fname, "_pic_gs.jpg");
	rotate_picture_test(GPU_SIMPLE, fname, "_pic_g.jpg");
	rotate_picture_test(CPU, fname, "_pic_c.jpg");

	std::cout << "Check..." << std::endl;
	ASSERT(diff_img("_pic_c.jpg", "_pic_gs.jpg"), "Images are differ");
	ASSERT(diff_img("_pic_c.jpg", "_pic_g.jpg"), "Images are differ");
	std::cout << "Ok!" << std::endl;

	return 0;
}
