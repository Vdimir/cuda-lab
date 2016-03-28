#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#define CUDA_TIMER_START(id) cudaEvent_t start##id, stop##id; \
	    cudaEventCreate(&start##id); \
	    cudaEventCreate(&stop##id); \
	    cudaEventRecord(start##id, 0)

#define CUDA_TIMER_STOP(id, elapsedTime) cudaEventRecord(stop##id, 0); \
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

void rotate_array_cpu(uchar *source, uchar *target, int m, int n) {

	float elapsedTimeCPU = -1;
    clock_t startCPU = clock();

	for (int i = 0; i < m; ++i) {
		for (int j = n-1; j >=0; j--) {
	      	int SIdx = (i*n + (n-1 - j)) * COMP_CNT;
	       	int DIdx = (j*m + i) * COMP_CNT;

	       	*((uchar3*)&target[DIdx]) = *((uchar3*)&source[SIdx]);
		}
	}

    elapsedTimeCPU = (float)(clock()-startCPU)/CLOCKS_PER_SEC;

    std::cout << "CPU time: "<< elapsedTimeCPU << std::endl;
    return;
}

__global__ void rotate_gpu_kernel(const uchar *src, uchar *trg, int m, int n) {
	int tx = threadIdx.x,
			ty = threadIdx.y;
	int bx = blockIdx.x*blockDim.x,
			by = blockIdx.y*blockDim.y;
    int i = tx + bx;
    int j = ty + by;

    if (i >= m || j >= n) return;

 	int SIdx = (i*n + (n-1 - j)) * COMP_CNT;
 	int DIdx = (j*m + i) * COMP_CNT;

   	for (int c = 0; c < COMP_CNT; c++)
   		trg[DIdx+c] = src[SIdx+c];
}


#define TILE_W 128
#define TILE_H 128

__global__ void rotate_gpu_kernel_shared(uchar *src, uchar *trg, int m, int n) {
	int tx = threadIdx.x,
			ty = threadIdx.y;
	int bx = blockIdx.x*blockDim.x,
			by = blockIdx.y*blockDim.y;

    __shared__ uchar3 block[TILE_W][TILE_H];

    for (int i = 0; i < TILE_W/blockDim.x; ++i) {
        int SIdx = (tx+bx+i) + (ty+by)* TILE_H;
        block[ty][tx+i] = ((uchar3*)src)[SIdx];

	}

    __syncthreads();

    for (int i = 0; i < TILE_W/blockDim.x; ++i) {
        int SIdx = (tx+bx+i) + (ty+by)* TILE_H;
        ((uchar3*)trg)[SIdx] = block[ty][tx+i];
	}


}


void call_kernel(uchar *dev_src, uchar *dev_res, int m, int n) {
	const int bsx = 8, bsy = 128;
    dim3 threadsPerBlock(bsx, bsy);
    dim3 numBlocks((m + (bsx-1)) / bsx, (n + (bsy-1)) / bsy);

    rotate_gpu_kernel_shared<<<numBlocks, threadsPerBlock >>>(dev_src, dev_res, m, n);
    //rotate_gpu_kernel<<<numBlocks, threadsPerBlock >>>(dev_src, dev_res, m, n);
}


void rotate_array_gpu(uchar *h_src, uchar *h_res,  int m, int n) {

	int N = m*n*COMP_CNT;
    uchar *dev_src;
    uchar *dev_res;


    cudaMalloc(&dev_src, N);
    cudaMalloc(&dev_res, N);

    CHECK(cudaMemcpy(dev_src, h_src, N, cudaMemcpyHostToDevice));

    CUDA_TIMER_START(A);

    call_kernel(dev_src, dev_res, m, n);

    float elapsedTime = -1;
    CUDA_TIMER_STOP(A, elapsedTime);

    std::cout << "CUDA time: "<< elapsedTime << std::endl;

    CHECK(cudaMemcpy(h_res, dev_res, N, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dev_src));
    CHECK(cudaFree(dev_res));
    return;
}

template<typename F>
void rotate_picture_test(F rotate_proc, const char * in_file_name, const char * out_file_name) {
	cv::Mat image = cv::imread(in_file_name, CV_LOAD_IMAGE_COLOR);

	if (!image.data) {
		std::cout << "Error open image" << std::endl;
		exit(1);
	}
	int n = image.cols,
			m = image.rows;

	//cv::Mat rotated(n, m, CV_8UC3);
	cv::Mat rotated(m, n, CV_8UC3);
	rotate_proc(image.data, rotated.data, m, n);

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

int main() {
//	rotate_picture_test(rotate_array_cpu, "pic.jpg", "_pic3.jpg");
	rotate_picture_test(rotate_array_gpu, "pic.jpg", "_pic2.jpg");

	//ASSERT(diff_img("_pic2.jpg", "_pic3.jpg"), "Images are differ");

	return 0;
}
