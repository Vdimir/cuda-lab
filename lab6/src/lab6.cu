
#include <iostream>
#include <opencv2/core/core.hpp>
//#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#include <sstream>

#define CHECK(value) {													   \
    cudaError_t _m_cudaStat = value;                                       \
    if (_m_cudaStat != cudaSuccess) {                                      \
        std::cerr<< "Error:" << cudaGetErrorString(_m_cudaStat) 		   \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                           \
    } }



#define MAX_PACK_SIZE 100

#define MEAN_STORAGE_SIZE 1

__device__ __host__ inline int idx3(const int x, const int y, const int z, int m, int n) {
     return x*m*n + y*n+z;
}

__device__ __host__ inline int d_uchar3(const uchar3 a, const uchar3 b) {

	return max(
			max(
				abs(a.x - b.x),
				abs(a.y - b.y)),
			abs(a.z - b.z));
}


#define IDX3(x,y,z) ((x)*m*n + (y)*n+(z))

__global__ void process_frames(int frames_count, const uchar3 *frames,
									uchar3 *mean_storage, int* mean_weigth,
									int m, int n)
{
	int tx = threadIdx.x,
			ty = threadIdx.y;
	int bx = blockIdx.x*blockDim.x,
			by = blockIdx.y*blockDim.y;
    int i = tx + bx;
    int j = ty + by;

    if (i >= m || j >= n) return;


    uchar3 opt_mean;
    int opt_weight = 0;


    const int eps = 15;

    int k = 0;
    while (k < frames_count) {
    	//int cur_idx = k;
		uchar3 mean = frames[IDX3(k,i,j)];
		int weight = 1;

		k++;
		while (d_uchar3(mean, frames[IDX3((k+2) % frames_count,i,j)]) < eps) {
			weight++;
			k++;
			if (k >= frames_count) { break; }
		}
		if (weight > opt_weight) {
			opt_mean = mean;
			opt_weight = weight;
		}
    }

    for (k = 0; k < MEAN_STORAGE_SIZE; ++k) {
    	int sidx = k*m*n+i*n+j;
        if (mean_weigth[sidx] == 0) {
            mean_storage[sidx] = opt_mean;
    		mean_weigth[sidx] = opt_weight;
    		break;
        }
        if (d_uchar3(mean_storage[sidx], opt_mean) < eps ) {
    		mean_weigth[sidx] += opt_weight;
    		/*int kk = k;
    		while ((--kk) > 0) {
    	    	int prev_sidx = kk*m*n+i*n+j;
    			if (mean_weigth[sidx] > mean_weigth[prev_sidx]) {
    				int t; uchar3 ut;
    				t = mean_weigth[sidx];
    				mean_weigth[sidx] = mean_weigth[prev_sidx];
    				mean_weigth[prev_sidx] = t;

    				ut = mean_storage[sidx];
    				mean_storage[sidx] = mean_storage[prev_sidx];
    				mean_storage[prev_sidx] = ut;
    			} else { break; }
    		}*/
        	break;
        }
        if (mean_weigth[i*n+j] < opt_weight) {
    		mean_storage[i*n+j] = opt_mean;
    		mean_weigth[i*n+j] = opt_weight;
    		break;
    	}
	}



    return;
    if (mean_weigth[i*n+j] == 0) {
        mean_storage[i*n+j] = opt_mean;
		mean_weigth[i*n+j] = opt_weight;
    } else {
    	if (d_uchar3(mean_storage[i*n+j], opt_mean) < eps ) {
    		mean_weigth[i*n+j] += opt_weight;
    	} else if (mean_weigth[i*n+j] < opt_weight) {
    		mean_storage[i*n+j] = opt_mean;
    		mean_weigth[i*n+j] = opt_weight;
    	}
    }
/*
    int sumx = 0, sumy = 0, sumz = 0;
    int sum_cout = 0;
    for (int k = 0; k < frames_count; ++k) {
    	if (abs(frames[IDX3(k,i,j)].x-frames[IDX3((k+1)%frames_count,i,j)].x) < 5 &&
    			abs(frames[IDX3(k,i,j)].y-frames[IDX3((k+1)%frames_count,i,j)].y) < 5&&
    			abs(frames[IDX3(k,i,j)].z-frames[IDX3((k+1)%frames_count,i,j)].z) < 5) {
        	sumx += frames[IDX3(k,i,j)].x;
        	sumy += frames[IDX3(k,i,j)].y;
        	sumz += frames[IDX3(k,i,j)].z;
        	sum_cout++;
    	}
	}

    uchar3 mean;

    if (sum_cout > 0) {
		mean.x = sumx/sum_cout;
		mean.y = sumy/sum_cout;
		mean.z = sumz/sum_cout;
    } else {
		mean.x = 255;
		mean.y = 0;
		mean.z = 0;

    }

 //   mean_storage[i*n+j] = mean;
  //  return;

    if (mean_weigth[i*n+j] == 0) {
        mean_storage[i*n+j] = mean;
    } else {
    	mean_storage[i*n+j].x = (mean_storage[i*n+j].x + mean.x)/2;
    	mean_storage[i*n+j].y = (mean_storage[i*n+j].y + mean.y)/2;
    	mean_storage[i*n+j].z = (mean_storage[i*n+j].z + mean.z)/2;
    	mean_weigth[i*n+j] = 1;
    }
    */
}

class BackgroundExtractor {
public:
	BackgroundExtractor(int im_h, int im_w) :
		m(im_h),
		n(im_w),
		pack_size(0),
		pack_iter(0)
	{

		int N = get_frame_size();

	    CHECK(cudaMalloc(&dev_mean_weight, sizeof(int)*im_h*im_w*MEAN_STORAGE_SIZE));
	    CHECK(cudaMemset(dev_mean_weight, 0, sizeof(int)*im_h*im_w*MEAN_STORAGE_SIZE));
	    CHECK(cudaMalloc(&dev_mean_storage, N*MEAN_STORAGE_SIZE));
	    CHECK(cudaMalloc(&dev_frames_pack, N*MAX_PACK_SIZE));

	    h_frames_pack = (uchar3*)malloc(N*MAX_PACK_SIZE);
	}

	~BackgroundExtractor() {
	    CHECK(cudaFree(dev_mean_weight));
	    CHECK(cudaFree(dev_mean_storage));
	    CHECK(cudaFree(dev_frames_pack));
	    free(h_frames_pack);
	}

	inline void add_to_current_pack(const uchar *frame) {

	    if (pack_size > MAX_PACK_SIZE) { std::cerr << "ERROR: Cant add image to pack" << std::endl; return; }

		int N = get_frame_size();

		memcpy(h_frames_pack + pack_size*m*n, frame, N);

	    std::cerr << "add_pack " << pack_size << " * " << pack_iter <<  std::endl;
	    (this->pack_size)++;
	}

	void process_pack() {
		const int bsx = 16, bsy = 16;
	    dim3 threadsPerBlock(bsx, bsy);
	    dim3 numBlocks((m + (bsx-1)) / bsx, (n + (bsy-1)) / bsy);

		int N = get_frame_size();

		CHECK(cudaMemcpy(dev_frames_pack, h_frames_pack, N*pack_size, cudaMemcpyHostToDevice));

		process_frames<<<numBlocks, threadsPerBlock>>>(pack_size,
														(const uchar3 *)dev_frames_pack,
														dev_mean_storage,
														dev_mean_weight,
														m, n);

		pack_size = 0;
		pack_iter++;
	}

	inline uchar* get_background_image() {
		int N = get_frame_size();
		uchar* res = new uchar[N];

		CHECK(cudaMemcpy(res, dev_mean_storage, N, cudaMemcpyDeviceToHost));
	    return res;
	}

private:

	int get_frame_size() {
		int N = m*n*3;
		return N;
	}

	uchar3 *dev_mean_storage;
	int* dev_mean_weight;
	uchar3 *h_frames_pack;
	uchar3 *dev_frames_pack;
	//const uchar3 *h_frames_pack[MAX_PACK_SIZE];

	int m, n;
	int pack_size;
	int pack_iter;
};




int main(int argc, char *argv[])
{
	VideoCapture cap;

	int pass_first_frames = 0;
	if (argc <= 1){
		cap.open("MVI_9264.MOV");
		pass_first_frames = 60;
	} else {
		cap.open(argv[1]);

	}


    if(!cap.isOpened()){
    	std::cerr << " Could`t open video file!" << std::endl;
    	exit(1);
    }

    cv::Mat image;
    for (int i = 0; i < pass_first_frames; ++i) {
		cap >> image;
	}


    BackgroundExtractor ext(cap.get(CV_CAP_PROP_FRAME_HEIGHT), cap.get(CV_CAP_PROP_FRAME_WIDTH));

    int packN = 0;
    while (true) {

        for (int i = 0; i < MAX_PACK_SIZE; ++i) {
    		cap >> image;
        	if (image.empty()) { break; }

    		ext.add_to_current_pack(image.data);
    	}

        ext.process_pack();

        image.data = (uchar*)ext.get_background_image();


    	std::stringstream ss;
    	ss << "res" << packN++ << ".jpg";
        cv::imwrite(ss.str(), image);

    	if (image.empty()) { break; }
    }
    return 0;
}
