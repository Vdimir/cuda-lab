#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

using namespace std;
typedef unsigned char uchar;


#define CUDA_TIMER_START(id) cudaEvent_t start##id, stop##id; \
	    cudaEventCreate(&start##id); \
	    cudaEventCreate(&stop##id); \
	    cudaEventRecord(start##id, 0)

#define CUDA_TIMER_STOP(id, elapsedTime) cudaEventRecord(stop##id, 0); \
	    cudaEventSynchronize(stop##id); \
	    cudaEventElapsedTime(&elapsedTime, start##id, stop##id); \
	    cudaEventDestroy(start##id); \
	    cudaEventDestroy(stop##id)

template<typename T>
void display_matrix(const T& v, int col_count, int row_count) {
	for (int i = 0; i < col_count; ++i) {
		for (int j = 0; j < row_count; ++j) {
			cout.width(4);
			cout << v[i*row_count+j];
		}
		cout << endl;
	}
}

void rotate_array_cpu(uchar *source, uchar *target, int m, int n, int comp_count=1) {

	float elapsedTimeCPU = 0;

    CUDA_TIMER_START(B);
    //clock_t startCPU = clock();

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
	      	int SIdx = (i*n + (n-1 - j)) * comp_count;
	       	int DIdx = (j*m + i) * comp_count;

	       	for (int c = 0; c < comp_count; ++c)
	       		target[DIdx+c] = source[SIdx+c];
		}
	}

    //elapsedTimeCPU = (float)(clock()-startCPU)/CLOCKS_PER_SEC;
    CUDA_TIMER_STOP(B, elapsedTimeCPU);
    std::cout << "CPU time: "<< elapsedTimeCPU << std::endl;
    return;
}

__global__ void rotate_gpu_kernel(uchar *src, uchar *trg, int m, int n, int comp_count) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    if (i >= m || j >= n) return;

 	int SIdx = (i*n + (n-1 - j)) * comp_count;
 	int DIdx = (j*m + i) * comp_count;

   	for (int c = 0; c < comp_count; c++)
   		trg[DIdx+c] = src[SIdx+c];
}

void call_kernel(uchar *dev_src, uchar *dev_res, int m, int n, int comp_count) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m+15) / 16, (n+15) / 16);

    rotate_gpu_kernel<<<numBlocks, threadsPerBlock >>>(dev_src, dev_res, m, n, comp_count);
}


void rotate_array_gpu(uchar *h_src, uchar *h_res,  int m, int n, int comp_count) {

	int N = m*n*comp_count;
    unsigned char *dev_src;
    unsigned char *dev_res;

    float elapsedTime;

    cudaMalloc(&dev_src, N);
    cudaMalloc(&dev_res, N);

    cudaMemcpy(dev_src, h_src, N, cudaMemcpyHostToDevice);

    CUDA_TIMER_START(A);
    //clock_t startCPU = clock();
    call_kernel(dev_src, dev_res, m, n, comp_count);

    CUDA_TIMER_STOP(A, elapsedTime);
    //elapsedTime = (float)(clock()-startCPU)/CLOCKS_PER_SEC;

    cudaMemcpy(h_res, dev_res, N, cudaMemcpyDeviceToHost);
    cudaFree(dev_src);
    cudaFree(dev_res);

    std::cout << "CUDA time: "<< elapsedTime << std::endl;
    return;
}

template<typename F>
void rotate_picture_test(F rotate, const char * in_file_name, const char * out_file_name) {
	cv::Mat image = cv::imread(in_file_name, CV_LOAD_IMAGE_COLOR);

	int n = image.cols,
			m = image.rows;
	cv::Mat rotated(n, m, CV_8UC3);

	rotate(image.data, rotated.data, m, n, 3);

    cv::imwrite(out_file_name,rotated);
    return;
}


/*
void rotate_vector_test() {
	const int m = 3;
	const int n = 4;
	const int array_size = m*n;
	vector<int> source(array_size);
	for (int i = 0; i < m*n; ++i) { source[i] = i+1; }

	vector<int> target(array_size);

	rotate_array_cpu(source, target, m, n);

	display_matrix(source, m, n);
	cout << endl;
	display_matrix(target, n, m);
}
*/

int main() {
	rotate_picture_test(rotate_array_gpu, "pic.jpg", "_pic2.jpg");
	rotate_picture_test(rotate_array_cpu, "pic.jpg", "_pic3.jpg");

	return 0;
}

