#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define checkError(func) \
  if (errcode != CL_SUCCESS)\
  {\
    printf("Error in " #func "\nError code = %d\n", errcode);\
    exit(1);\
  }

#define checkErrorEx(command) \
  command; \
  checkError(command);


//GLOBAL VARIABLES
cl_context context;
cl_kernel kernel;
cl_command_queue queue;

#define BYTE_PER_PIXEL 3


void get_platforms(cl_platform_id platform[]) {
	  cl_int errcode;

	  cl_uint num_platforms;
	  errcode = clGetPlatformIDs(10, platform, &num_platforms);
	  checkError(clGetPlatformIDs);
	  printf("OpenCL platforms found: %d\n", num_platforms);
	  return;
}

void get_devices(cl_platform_id platform, cl_device_id *devices) {
	  cl_uint num_devices;
	  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 10, devices, &num_devices);
	  printf("GPGPU devices found: %d\n", num_devices);
	  if (num_devices == 0)
	  {
	      printf("Error: YOU DON'T HAVE GPGPU.\n");
	      exit(1);
	  }
}

void show_device_info(cl_device_id device) {
	   size_t valueSize;
	   clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
	   char* value = (char*) malloc(valueSize);
	   clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, value, NULL);
	   printf("Use device %s\n", value);
	   free(value);

}
void build_cl_program(cl_program *program, cl_device_id *device) {

	cl_int errcode = clBuildProgram(*program, 1, device,
			"-cl-fast-relaxed-math -cl-no-signed-zeros -cl-mad-enable", NULL,
			NULL);
	if (errcode != CL_SUCCESS) {
		size_t len;
		char *build_log;
		printf(
				"There were error during build kernel code. Please, check program code. Errcode = %d\n",
				errcode);
		clGetProgramBuildInfo(*program, *device, CL_PROGRAM_BUILD_LOG, 0, NULL,
				&len);
		build_log = (char *)malloc(len + 1);
		clGetProgramBuildInfo(*program, *device, CL_PROGRAM_BUILD_LOG, len,
				build_log, NULL);
		build_log[len] = '\0';
		printf("BUILD LOG: %s\n", build_log);
		exit(1);
	}

	kernel = clCreateKernel(*program, "sum", &errcode);
	checkError(clCreateKernel);
	return;
}

void cl_initilize() {
	cl_int errcode;

	cl_platform_id platform[10];
	get_platforms(platform);

	cl_device_id devices[10];
	int platform_index = 0;
	get_devices(platform[platform_index], devices);

	int device_index = 0;
	show_device_info(devices[device_index]);

	context = clCreateContext(NULL, 1, &devices[device_index], NULL, NULL, &errcode);
	checkError(clCreateContext);

	queue = clCreateCommandQueue (context, devices[device_index], CL_QUEUE_PROFILING_ENABLE, &errcode); // третий параметр - свойства
	checkError(clCreateCommandQueue);

	char* source = "\n\
			  __kernel void sum(__global const uchar *src, __global uchar *trg, int m, int n)\n\
			  {\n\
			    int  i = get_global_id(0);\n\
			    int  j = get_global_id(1);\n\
 	            int SIdx = (i*n + (n-1 - j)) ;\n\
		    	int DIdx = (j*m + i) ;\n\
				if (i > m) return;\
				if (j > n) return;\
            	for (int c = 0; c < 3; c++)\n\
   		            trg[DIdx*3+c] = src[SIdx*3+c];\n\
			  }";
	cl_program program = clCreateProgramWithSource(context, 1, (const char **) &source, NULL, &errcode);
	checkError(clCreateProgramWithSource);

	build_cl_program(&program, &devices[device_index]);

}


float rotate_array_gpu(uchar *h_src, uchar *h_res,  int m, int n) {

	int N = m*n*BYTE_PER_PIXEL;

	cl_int errcode;

	  cl_mem dev_src, dev_res;
	  checkErrorEx(dev_src = clCreateBuffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N*sizeof(uchar), h_src, &errcode ));
	  checkErrorEx(dev_res = clCreateBuffer( context, CL_MEM_READ_WRITE, N*sizeof(uchar), NULL, &errcode ));

	  checkErrorEx(errcode = clSetKernelArg(kernel, 0, sizeof(dev_src), &dev_src););
	  checkErrorEx(errcode = clSetKernelArg(kernel, 1, sizeof(dev_res), &dev_res););
	  checkErrorEx(errcode = clSetKernelArg(kernel, 2, sizeof(m), &m););
	  checkErrorEx(errcode = clSetKernelArg(kernel, 3, sizeof(n), &n););


	const int bsx = 16, bsy = 16;

	  size_t globalSize[] = {8192, 4096};
	  printf("%d\n", globalSize[0]);
	  size_t locallSize[] = {bsx, bsy};
	   // ставим задачу в очередь.
	   // 3й аргумент - размерность пространства рабочих,
	   // 6й аргумент - размер work-group
	   // пследние 3 аргумента для событий
	   clock_t t0 = clock();
	   cl_event event;
	   errcode = clEnqueueNDRangeKernel(queue,		// cl_command_queue command_queue,		
	   								kernel,			// cl_kernel kernel,	
	   								2,				// cl_uint work_dim,
	   								NULL,			// const size_t *global_work_offset,	
	   								globalSize,	// const size_t *global_work_size,			
	   								locallSize,			// const size_t *local_work_size,	
	   								0,				// cl_uint num_events_in_wait_list,
	   								NULL,			// const cl_event *event_wait_list,	
	   								&event);		// cl_event *event)		
	   checkError(clEnqueueNDRangeKernel);
	   checkErrorEx(errcode = clWaitForEvents(1, &event));
	   clock_t t1 = clock();
	   cl_ulong time_start, time_end;
	   checkErrorEx(errcode = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL));
	   checkErrorEx(errcode = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL));
	   double elapsedTimeGPU;
	   if (errcode == CL_PROFILING_INFO_NOT_AVAILABLE) elapsedTimeGPU = (double)(t1-t0)/CLOCKS_PER_SEC;
	   else elapsedTimeGPU = (double)(time_end - time_start)/1e9;
	   clReleaseEvent(event);

	   checkErrorEx(errcode = clEnqueueReadBuffer( queue, dev_res, CL_TRUE, 0, N*sizeof(uchar), h_res, 0, NULL, NULL));
	   checkErrorEx(errcode = clReleaseMemObject(dev_src));
	   checkErrorEx(errcode = clReleaseMemObject(dev_res));

	   printf( "GPU time = %g ms\n", elapsedTimeGPU*1000);

    return 0.0f;
}


void rotate_picture(const char * in_file_name, const char * out_file_name) {
	cv::Mat image = cv::imread(in_file_name, CV_LOAD_IMAGE_COLOR);

	if (!image.data) {
		std::cout << "Error open image" << std::endl;
		exit(1);
	}
	int n = image.cols,
			m = image.rows;

	cv::Mat rotated(n, m, CV_8UC3);

	rotate_array_gpu(image.data, rotated.data, m, n);

    cv::imwrite(out_file_name, rotated);
    return;
}

int main()
{
	cl_initilize();
	rotate_picture("pic.jpg", "_pic_g.jpg");

	return 0;
}
