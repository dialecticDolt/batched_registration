//CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_fp16.h"
#include <cuComplex.h>

//NPP
#include<npp.h>
#include<nppi_geometry_transforms.h>
#include<nppdefs.h>

//CUFFT
#include<cufft.h>
#include<cufftXt.h>

//OPENCV
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>

//THRUST
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/extrema.h>
#include<thrust/sort.h>
#include<thrust/sequence.h>
#include<thrust/device_malloc.h>
#include<thrust/device_free.h>
#include<thrust/iterator/transform_iterator.h>
#include<thrust/iterator/counting_iterator.h>
#include<thrust/iterator/discard_iterator.h>
#include<thrust/execution_policy.h>
#include<thrust/reduce.h>

//CPP
#include<stdio.h>
#include<iostream>
#include<chrono>
#include<vector>
#include<cmath>
#include<assert.h>
#include<stdexcept>
#include<sstream>
#include<type_traits>
#include<fstream>
#include<cstdlib>
#include<utility>

#include "helper_cuda.h"
#include "convolutionSeparable.h"

#include "singleton.h"
#include "phaseCorrelation.h"
#include "profiler.h"

#include "complexOps.h"
#include "util.h"

#include <thread>


using namespace thrust::placeholders;

int testfunc() {
	return 4;
}


template<typename T>
class DualData {
public:
	T* host_data;
	T* device_data = NULL;
	bool hstate = false;
	bool dstate = false;

	size_t total_size = 0;
	size_t active_size = 0;

	size_t total_bytes = 0;
	size_t active_bytes = 0;

	std::string name = "Unnamed";

	DualData() {};

	DualData(std::string name, cv::Mat Image) {
		cv::Size shape = Image.size();
		assert(Image.channels() == 1);
		size_t image_size = shape.area();

		this->total_size = image_size;
		this->active_size = image_size;

		this->active_bytes = image_size * sizeof(T);
		this->total_bytes = image_size * sizeof(T);

		this->host_data = static_cast<T*>(malloc(this->total_bytes));
		std::memcpy(this->host_data, static_cast<T*>(Image.data), this->active_bytes);
		this->hstate = true;

		this->name = name;
	}

	DualData(T* h_data, T* d_data, size_t size) {
		this->host_data = h_data;
		this->hstate = true;

		this->device_data = d_data;
		this->dstate = true;

		this->total_size = size;
		this->active_size = size;
	}

	DualData(T* h_data, T* d_data, size_t active_size, size_t total_size) {
		this->host_data = h_data;
		this->hstate = true;

		this->device_data = d_data;
		this->dstate = true;

		this->active_size = active_size;
		this->total_size = total_size;

		this->active_bytes = active_size * sizeof(T);
		this->total_bytes = total_size * sizeof(T);
	}

	DualData(std::string name, T* data, size_t total_size, std::string type) {
		if (type == "HOST") {
			this->host_data = data;
			this->hstate = true;
		}
		else if (type == "DEVICE") {
			this->device_data = data;
			this->dstate = true;
		}

		this->total_size = total_size;
		this->active_size = total_size;

		this->active_bytes = total_size * sizeof(T);
		this->total_bytes = total_size * sizeof(T);
		this->name = name;
	}

	DualData(T* data, size_t active_size, size_t total_size, std::string type) {
		if (type == "HOST") {
			this->host_data = data;
			this->hstate = true;
		}
		else if (type == "DEVICE") {
			this->device_data = data;
			this->dstate = true;
		}

		this->total_size = total_size;
		this->active_size = active_size;

		this->active_bytes = active_size * sizeof(T);
		this->total_bytes = total_size * sizeof(T);
	}

	DualData(std::string name, size_t total_size) {
		this->active_size = total_size;
		this->total_size = total_size;

		this->active_bytes = total_size * sizeof(T);
		this->total_bytes = total_size * sizeof(T);

		this->name = name;
	}

	void setDevice(T* ddata, size_t total_size) {
		if (!(this->hstate || this->dstate)) {
			this->device_data = ddata;
			this->dstate = true;

			this->active_size = total_size;
			this->total_size = total_size;

			this->active_bytes = total_size * sizeof(T);
			this->total_bytes = total_size * sizeof(T);
		}
		else {
			throw std::runtime_error(Formatter() << "setDevice: Cannot set after either Host or Device data has been initialized");
		}
	}

	void setDevice(T* ddata, size_t active_size, size_t total_size) {
		if (!(this->hstate || this->dstate)) {
			this->host_data = ddata;
			this->dstate = true;

			this->active_size = active_size;
			this->total_size = total_size;

			this->active_bytes = active_size * sizeof(T);
			this->total_bytes = total_size * sizeof(T);
		}
		else {
			throw std::runtime_error(Formatter() << "setDevice: Cannot set after either Host or Device data has been initialized");
		}
	}

	void setHost(T* hdata, size_t total_size) {
		if (!(this->hstate || this->dstate)) {
			this->host_data = hdata;
			this->hstate = true;

			this->active_size = total_size;
			this->total_size = total_size;

			this->active_bytes = total_size * sizeof(T);
			this->total_bytes = total_size * sizeof(T);
		}
		else {
			throw std::runtime_error(Formatter() << "setHost: Cannot set after either Host or Device data has been initialized");
		}
	}

	void setHost(T* hdata, size_t active_size, size_t total_size) {
		if (!(this->hstate || this->dstate)) {
			this->host_data = hdata;
			this->hstate = true;

			this->active_size = active_size;
			this->total_size = total_size;

			this->active_bytes = active_size * sizeof(T);
			this->total_bytes = total_size * sizeof(T);
		}
		else {
			throw std::runtime_error(Formatter() << "setHost: Cannot set after either Host or Device data has been initialized");
		}
	}

	void createHost() {
		if (!this->hstate) {
			this->host_data = static_cast<T*>(malloc(this->total_bytes));
			this->hstate = true;
			std::cout << name << " has been allocated." << std::endl;
		}
		else {
			std::cout << name << " has already been set!" << std::endl;
		}

	}

	void createDevice() {
		if (!this->dstate) {
			CUDA_CHECK(cudaMalloc((void**)&this->device_data, this->total_bytes));
			this->dstate = true;
			std::cout << name << " has been allocated." << std::endl;
		}
		else {
			std::cout << name << " has already been set!" << std::endl;
		}
	}

	void pull() {
		if (this->dstate) {
			if (!this->hstate) {
				this->createHost();
			}
			CUDA_CHECK(cudaMemcpy(this->host_data, this->device_data, this->total_bytes, cudaMemcpyDeviceToHost));
		}
		else {
			std::cout << "WARNING" << std::endl;
		}
	}

	void push() {
		if (this->hstate) {
			if (!this->dstate) {
				this->createDevice();
			}
			CUDA_CHECK(cudaMemcpy(this->device_data, this->host_data, this->total_bytes, cudaMemcpyHostToDevice));
		}
		else {
			std::cout << "WARNING" << std::endl;
		}
}

	template<typename N>
	DualData<N> convertGPU() {
		size_t old_unit_size = sizeof(T);
		size_t new_unit_size = sizeof(N);
		if (new_unit_size * this->active_size > this->total_bytes) {
			throw std::runtime_error(Formatter() << "Failed to convert. New type requires " << new_unit_size * this->active_size*new_unit_size << " bytes. Only  " << this->total_bytes << " bytes have been allocated");
		}

		N* host_data = reinterpret_cast<N*>(this->host_data);
		N* device_data = reinterpret_cast<N*>(this->host_data);

		//PUT A CUDA KERNEL HERE TO CONVERT

		return DualData(host_data, device_data, this->active_size, this->total_size);
	}

	void destroyDevice() {
		if (this->dstate) {
			this->dstate = false;
			CUDA_CHECK( cudaFree(this->device_data) );
			this->device_data = NULL;
		}
	}

	void destroyHost() {
		if (this->hstate) {
			this->hstate = false;
			free(this->host_data);
			this->host_data = NULL;
		}
	}

	void clean() {
		this->destroyDevice();
		this->destroyHost();
	}

	~DualData() {
		//this->clean();
	}
};

class Parameter {
public:
	Npp64f start = 0;
	Npp64f end = 0;
	unsigned int N = 0;
	Npp64f delta = 0;
	unsigned int initial = 0;
	Npp64f result = -1.0;

	Parameter(Npp64f start, Npp64f end, unsigned int N) {
		this->start = start;
		this->end = end;
		this->N = N;
		if (N == 1) {
			N = 2;
		}
		this->delta = (end - start) / (N-1);
		this->initial = 0;
	}
	Parameter() = default;

	~Parameter() {};

	void increment(unsigned int k) {
		this->start = this->start + k * this->delta;
	}
};

class Result {
public:
	Npp64f angle = 0;
	Npp64f scale_h = 0;
	Npp64f scale_w = 0;

	Npp64f dh = 0;
	Npp64f dw = 0;

	Npp64f value = 0;

	Result(Npp64f angle, Npp64f scale_h, Npp64f scale_w, Npp64f value, Npp64f dh = 0, Npp64f dw = 0) {
		this->angle = angle;
		this->scale_w = scale_w;
		this->scale_h = scale_h;
		this->value = value;
		this->dh = dh;
		this->dw = dw;
	}

	Result() = default;

	~Result() {};
};

bool inline operator<(Result a, Result b) { return a.value < b.value; };
bool inline operator>(Result a, Result b) { return a.value > b.value; };

class SearchContext {
public:
	unsigned int k = 5;
	Parameter theta = Parameter(0, 0, 0);
	Parameter scale_w = Parameter(0, 0, 0);
	Parameter scale_h = Parameter(0, 0, 0);
	float value = 0;
	std::vector<Result> topK;

	SearchContext(Parameter theta, Parameter scale_w, Parameter scale_h, unsigned int k=5) {
		this->theta = theta;
		this->scale_w = scale_w;
		this->scale_h = scale_h;
		this->k = k;
		topK.resize(k);
	}

	SearchContext() = default;
	~SearchContext() {};
};


template<typename T>
class ReferenceContext {
public:
	DualData<T> image;

	NppiRect box;
	DualData<double> quad = DualData<double>("Ref Quad", 8);
	cv::Size image_shape;
	double area = 0.0;

	ReferenceContext() {};

	ReferenceContext(DualData<T>& image, cv::Size shape) {
		this->image = image;
		this->image_shape = shape;
	}
};

template<typename T>
class ReferenceContextPC : public ReferenceContext<T> {
public:

	DualData<cufftComplex> workspace;

	ReferenceContextPC() {};

	ReferenceContextPC(DualData<T>& image, cv::Size shape) : ReferenceContext(image, shape) {
		//Allocate space for F[gradient]
		this->workspace = DualData<cufftComplex>("Reference Workspace", shape.area());
	}

};

template<typename T>
class ReferenceContextNGF : public ReferenceContext<T> {

public:

	DualData<cufftComplex> dx_workspace;
	DualData<cufftComplex> dy_workspace;
	DualData<cufftComplex> mix_workspace;

	ReferenceContextNGF() {};

	ReferenceContextNGF(DualData<T>& image, cv::Size shape) : ReferenceContext(image, shape) {
		//Allocate space for F[gx^2], F[gy^2], and F[mix]
		//This overallocates in the current implemenation 
		//TODO (fix this)
		size_t ws_size = shape.area();// ((size_t)(floor(shape.height / 2)) + 1)* shape.width;
		this->dx_workspace = DualData<cufftComplex>("Reference Dx Workspace", ws_size);
		this->dy_workspace = DualData<cufftComplex>("Reference Dy Workspace", ws_size);
		this->mix_workspace = DualData<cufftComplex>("Reference Mix Workspace", ws_size);
	}

};

template<typename T>
class WarpContext {
public:
	DualData<T> image;
	DualData<T> batch;

	NppiRect box;
	cv::Size image_shape;
	unsigned int nBatch = 0;

	DualData<Npp64f> transformBuffer;
	DualData<NppiWarpAffineBatchCXR> batchList;
	DualData<Npp64f> bboxBuffer;

	WarpContext() {};

	WarpContext(DualData<T>& image, cv::Size shape, unsigned int nBatch) {
		this->image = image;
		this->image_shape = shape;
		this->nBatch = nBatch;

		size_t transform_buffer_size = 6 * nBatch;
		this->transformBuffer = DualData<Npp64f>("Warp Context: transformBuffer", transform_buffer_size);

		size_t list_buffer_size = nBatch;
		this->batchList = DualData<NppiWarpAffineBatchCXR>("Warp Context: batchList", list_buffer_size);

		size_t batch_buffer_size = 2 * shape.area() * nBatch;
		this->batch = DualData<T>("Warp Context: batch", batch_buffer_size);

		size_t bbox_buffer_size = 8 * nBatch;
		this->bboxBuffer = DualData<Npp64f>("Warp Context: bboxBuffer", bbox_buffer_size);
	};

	~WarpContext() {};

};

template<typename T>
class WarpContextPC : public WarpContext<T> {
public:

	DualData<cufftComplex> workspace;

	WarpContextPC() {};

	WarpContextPC(DualData<T>& image, cv::Size shape, unsigned int nBatch) : WarpContext(image, shape, nBatch) {
		size_t ws_buffer_size = nBatch * shape.area();
		this->workspace = DualData<cufftComplex>("Warp Context: Complex Workspace", ws_buffer_size);
	};
};

template<typename T>
class WarpContextNGF : public WarpContext<T> {
public:
	WarpContextNGF() {};

	DualData<cufftComplex> workspace_dx;
	DualData<cufftComplex> workspace_dy;
	DualData<cufftComplex> workspace_mix;

	WarpContextNGF(DualData<T>& image, cv::Size shape, unsigned int nBatch) : WarpContext(image, shape, nBatch) {
		//This overallocates in the current implementation
		//TODO: fix this
		size_t ws_buffer_size = nBatch * shape.area();// (size_t)nBatch* ((floor((shape.height) / 2)) + 1)* shape.width;
		this->workspace_dx = DualData<cufftComplex>("Warp Context: Dx Workspace", ws_buffer_size);
		this->workspace_dy = DualData<cufftComplex>("Warp Context: Dy Workspace", ws_buffer_size);
		this->workspace_mix = DualData<cufftComplex>("Warp Context: Mix Workspace", ws_buffer_size);
	};
};

template<typename T>
class RegistrationContext {
public:
	DualData<T> ReferenceImage;
	DualData<T> MovingImage;

	DualData<T> batchMoving;

	RegistrationContext() {};
	~RegistrationContext() {};

};


void showImage(cv::Mat &Image, std::string title) {
	cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
	cv::imshow(title, Image);
	cv::waitKey(0);
}

void setGradientKernel() {
	float* h_Kernel;
	h_Kernel = (float*)malloc(KERNEL_LENGTH * sizeof(float));
	if (h_Kernel == NULL) {
		throw  std::runtime_error("Failed to allocate memory for gradient kernel.");
	}

	h_Kernel[0] = float(1.0) / 12;
	h_Kernel[1] = -float(2.0) / 2;
	h_Kernel[2] = 0.0;
	h_Kernel[3] = float(2.0) / 3;
	h_Kernel[4] = float(1.0) / 12;

	setConvolutionKernel(h_Kernel);
}

template<typename T, typename N>
__global__ void copy_convert_kernel(T* __restrict__ output, N* __restrict__ input, const unsigned int n) {
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
		output[i] = (T)input[i];
	}
}

__global__ void repack_pc(cufftComplex* __restrict__ output, float* __restrict__ gx, float* __restrict__ gy, const unsigned int n) {
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {

		//Normalize and copy into workspaces for FFT
		const float eps = 0.00001;
		float norm = 1.0;// sqrt(gx[i] * gx[i] + gy[i] * gy[i] + eps);
		output[i].x = gx[i] / norm;
		output[i].y = gy[i] / norm;
	}
}

/*
__global__ void repack_ngf(float* __restrict__ dx_buffer, float* __restrict__ dy_buffer, float* __restrict__ mix_buffer, float* __restrict__ gx, float* __restrict__ gy, const unsigned int n) {
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {

		//Normalize and Regularize Gradients
		const float eps = 0.00001;
		float norm = 1.0; //sqrt(gx[i] * gx[i] + gy[i] * gy[i] + eps);

		//Copy into workspaces for FFT
		float temp_gx = gx[i] / norm;
		float temp_gy = gy[i] / norm;

		dx_buffer[i] = temp_gx * temp_gx;
		dy_buffer[i] = temp_gy * temp_gy;
		mix_buffer[i] = temp_gx * temp_gy;
	}
}
*/

__global__ void repack_ngf(cufftComplex* __restrict__ dx_buffer, cufftComplex* __restrict__ dy_buffer, cufftComplex* __restrict__ mix_buffer, float* __restrict__ gx, float* __restrict__ gy, const unsigned int n) {
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {

		//Normalize and Regularize Gradients
		const float eps = 0.01;
		float norm = sqrt( (gx[i] * gx[i]) + (gy[i] * gy[i]) + eps * eps);

		//Copy into workspaces for FFT
		float temp_gx = gx[i] / norm;
		float temp_gy = gy[i] / norm;

		dx_buffer[i].x = temp_gx * temp_gx;
		dy_buffer[i].x = temp_gy * temp_gy;
		mix_buffer[i].x = temp_gx * temp_gy;

		dx_buffer[i].y = 0;
		dy_buffer[i].y = 0;
		mix_buffer[i].y = 0;
	}
}

__global__ void pc_kernel(cufftComplex* __restrict__ output, cufftComplex* __restrict__ ref, const unsigned int n, const unsigned int chunk) {
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
		cufftComplex elref = ref[i % chunk];
		cufftComplex elmov = cong(output[i]);
		float magref = mag(elref);
		float magmov = mag(elmov);

		cufftComplex out;
		float thres = 0.1;
		out = ((elref / magref) * (elmov / magmov));
		out = magref > thres ? out : zero();
		out = magmov > thres ? out : zero();

		output[i] = out;
	}
}

__global__ void ngf_kernel(cufftComplex* dx_buffer, cufftComplex* dy_buffer, cufftComplex* mix_buffer, cufftComplex* gx_ref, cufftComplex* gy_ref, cufftComplex* mix_ref, const unsigned int n, const unsigned int chunk) {
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
		cufftComplex el_gx_ref = gx_ref[i % chunk]/chunk;
		cufftComplex el_gy_ref = gy_ref[i % chunk]/chunk;
		cufftComplex el_mix_ref = mix_ref[i % chunk]/chunk;

		cufftComplex el_gx_mov = cong(dx_buffer[i]);
		cufftComplex el_gy_mov = cong(dy_buffer[i]);
		cufftComplex el_mix_mov = cong(mix_buffer[i]);

		cufftComplex out;
		out = (el_gx_ref * el_gx_mov) + (el_gy_ref * el_gy_mov) + 2*(el_mix_ref * el_mix_mov);
		//out  = (el_gy_ref * el_gy_mov);
		//out = (el_gx_ref * el_gx_mov);
		//printf("( %f, %f ) \n", out.x, out.y);
		dx_buffer[i] = out;

	}
}

__global__ void normalize(cufftComplex* out, double* bbox_list, double ref_area, const unsigned int n, const unsigned int chunk) {
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
		unsigned int block = i / chunk;
		double* quad = bbox_list + block * 8;

		double a = area(quad);
		double b;
		//printf("Area ( %f, %f ) \n", ref_area, a);
		b = (a < ref_area) ? a : ref_area;
		//printf("Area ( %f, %f ) - %f \n", ref_area, a, b);
		out[i] = out[i] / a;
	}
}

/*
void image_reduce(Npp8u* warp, double* sum_list, const unsigned int nbatch, const unsigned int chunk) {
	thrust::device_ptr<unsigned char> d_warp = thrust::device_pointer_cast(warp);
	thrust::device_ptr<unsigned char> d_sum = thrust::device_pointer_cast(sum_list);

	auto count = thrust::make_transform_iterator(d_warp, d_warp + nbatch*chunk);
	auto keys_first = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1 / chunk);
	auto keys_last = thrust::make_transform_iterator(thrust::counting_iterator<int>(nbatch * chunk), _1 / chunk);

	thrust::reduce_by_key(thrust::device, keys_first, keys_last, d_warp, thrust::discard_iterator<int>(), d_sum);
}*/

std::pair<size_t, cufftComplex> argmax(cufftComplex* a, const unsigned int n) {
	thrust::device_ptr<cufftComplex> d_ptr = thrust::device_pointer_cast(a);
	thrust::device_vector<cufftComplex>::iterator iter = thrust::max_element(d_ptr, d_ptr + n, mag_comp());
	size_t position = thrust::device_pointer_cast( &(iter[0]) ) - d_ptr;
	cufftComplex corr = *iter;
	return std::pair<size_t, cufftComplex>(position, corr);
}

std::vector<std::pair<size_t, cufftComplex>> kselect_max(cufftComplex* a, const unsigned int n, const unsigned int k) {
	std::vector<std::pair<size_t, cufftComplex>> output(k);
	thrust::device_ptr<cufftComplex> d_ptr = thrust::device_pointer_cast(a);

	for (int i = 0; i < k; ++i) {

		thrust::device_vector<cufftComplex>::iterator iter = thrust::max_element(d_ptr, d_ptr + n, mag_comp());
		size_t position = thrust::device_pointer_cast(&(iter[0])) - d_ptr;
		cufftComplex corr = *iter;
		output[i] = std::make_pair(position, corr);
		d_ptr[position] = zero();

	}
	return output;
}

std::vector<std::pair<size_t, cufftComplex>> kselect_max(float* a, const unsigned int n, const unsigned int k) {
	std::vector<std::pair<size_t, cufftComplex>> output(k);
	thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(a);

	for (int i = 0; i < k; ++i) {

		thrust::device_vector<float>::iterator iter = thrust::max_element(d_ptr, d_ptr + n, comp());
		size_t position = thrust::device_pointer_cast(&(iter[0])) - d_ptr;
		cufftComplex  corr;
		corr.x = *iter;
		corr.y = 0;
		output[i] = std::make_pair(position, corr);
		d_ptr[position] = 0.0;

	}
	return output;
}


std::vector<std::pair<size_t, cufftComplex>> kselect(cufftComplex* a, const unsigned int n, const unsigned int k) {
	thrust::device_ptr<cufftComplex> d_ptr = thrust::device_pointer_cast(a);
	thrust::device_ptr<size_t> index_vector = thrust::device_malloc<size_t>(n);
	thrust::sequence(index_vector, index_vector+n);

	std::cout << index_vector[10] << std::endl;

	thrust::stable_sort_by_key(thrust::device, d_ptr, d_ptr + n, index_vector, thrust::greater<cufftComplex>());

	std::vector<std::pair<size_t, cufftComplex>> output(k);
	for (int i = 0; i < k; ++i) {
		output[i] = std::make_pair(index_vector[i], d_ptr[i]);
	}

	thrust::device_free(index_vector);
	return output;
}

void merge_k(std::vector<Result>& v1, std::vector<Result>& v2, const unsigned int k) {
	std::vector<Result> aux(2 * k);
	for (size_t i = 0; i < k; ++i) {
		aux[i] = v1[i];
		aux[i + k] = v2[i];
	}
	std::sort(aux.begin(), aux.end(), std::greater<Result>());

	for (size_t i = 0; i < k; ++i) {
		v1[i] = aux[i];
	}
}

void testArgMax() {
	unsigned int n = 512 * 512;
	DualData<cufftComplex> testData = DualData<cufftComplex>("Test", n);
	testData.createHost();

	for (int i = 0; i < n; ++i) {
		testData.host_data[i].x = 0;
		testData.host_data[i].y = 0;
		if (i == n/2) {
			testData.host_data[i].x = 100;
			testData.host_data[i].y = 100;
		}
		if (i == n / 4) {
			testData.host_data[i].x = 50;
			testData.host_data[i].y = 50;
		}
	}
	testData.push();

	//auto location_pair = argmax(testData.device_data, n);
	auto location_list = kselect(testData.device_data, n, 5);
	auto location_pair = location_list[0];

	std::cout << n / 2 << " != " << location_pair.first << std::endl;
	std::cout << n / 4 << " != " << location_list[1].first << std::endl;
	testData.clean();
}

void gradient(Npp8u* image, double* quad, cv::Size &shape, DualData<float> &buffer) {
	float* drows = buffer.device_data;
	float* dcols = buffer.device_data + ( shape.area() );

	assert(image != NULL);

	thrust::device_ptr<double> d_q = thrust::device_pointer_cast(quad);
	print("Quad", d_q, 8);

	//swapped width and height to see if it resolves ghosting bug 
	//CHECK
	convolutionColumnsGPU(dcols, image, shape.height, shape.width, quad);
	convolutionRowsGPU(drows, image, shape.height, shape.width, quad);

	
	//buffer.pull();

	//cv::Size dsize = cv::Size(shape.width, shape.height * 2);
	//auto test = toMat(buffer.host_data, dsize);
	//print("host_data", buffer.host_data, dsize.area());

	//showImage(test, "Gradient Images");
	
}


void gradient(WarpContextNGF<Npp8u>& image_data) {
	const unsigned int nBatch =image_data.nBatch;
	cv::Size shape = image_data.image_shape;

	assert(image_data.batch.dstate);

	int gridsize = 0, blocksize = 0;
	CUDA_CHECK( cudaOccupancyMaxPotentialBlockSize(&gridsize, &blocksize, repack_ngf, 0) );

	DualData<float> buffer = DualData<float>("Gradient Buffer", 2 * shape.area());
	buffer.createDevice();

	//float* dx_buffer = reinterpret_cast<float*>(image_data.workspace_dx.device_data);
	//float* dy_buffer = reinterpret_cast<float*>(image_data.workspace_dy.device_data);
	//float* mix_buffer = reinterpret_cast<float*>(image_data.workspace_mix.device_data);

	cufftComplex* dx_buffer = image_data.workspace_dx.device_data;
	cufftComplex* dy_buffer = image_data.workspace_dy.device_data;
	cufftComplex* mix_buffer = image_data.workspace_mix.device_data;


	for (size_t i = 0; i < nBatch; ++i) {
		Npp8u* image = image_data.batch.device_data + i * shape.area();
		cufftComplex* dx_out = dx_buffer + i * shape.area();
		cufftComplex* dy_out = dy_buffer + i * shape.area();
		cufftComplex* mix_out = mix_buffer + i * shape.area();

		double* quad = image_data.bboxBuffer.device_data + i * 8;
		gradient(image, quad, shape, buffer);

		float* gx = buffer.device_data;
		float* gy = buffer.device_data + (shape.area());
		repack_ngf<<<gridsize, blocksize>>>(dx_out, dy_out, mix_out, gx, gy, shape.area());
	}

	std::cout << "Recording Gradient :: " << shape.area() << std::endl; 
	image_data.workspace_dx.pull();
	
	writeBinary("dx.bin", image_data.workspace_dx.host_data, shape.area());

	image_data.workspace_dy.pull();
	writeBinary("dy.bin", image_data.workspace_dy.host_data, shape.area());

	image_data.workspace_mix.pull();
	writeBinary("mix.bin", image_data.workspace_mix.host_data, shape.area());

	buffer.destroyDevice();
}


void gradient(WarpContextPC<Npp8u> &image_data) {
	const unsigned int nBatch = image_data.nBatch;
	cv::Size shape = image_data.image_shape;

	assert(image_data.batch.dstate);

	int gridsize = 0, blocksize = 0;
	CUDA_CHECK( cudaOccupancyMaxPotentialBlockSize(&gridsize, &blocksize, repack_pc, 0) );

	DualData<float> buffer = DualData<float>("Gradient Buffer", 2 * shape.area());
	buffer.createDevice();
	//TODO: I changed this to a 1 to check the ghosting. Change it back
	for (size_t i = 0; i < nBatch; ++i) {
		Npp8u* image = image_data.batch.device_data + i * shape.area();
		cufftComplex* output = image_data.workspace.device_data + i * shape.area();

		double* quad = image_data.bboxBuffer.device_data + i * 8;
		gradient(image, quad, shape, buffer);
		std::cout << "Gradient shape: " << shape.width<< ", " << shape.height << std::endl;
		float* gx = buffer.device_data;
		float* gy = buffer.device_data + ( shape.area() );
		repack_pc<<<gridsize, blocksize>>>(output, gx, gy, shape.area());
	}
	buffer.pull();
	writeBinary("buffer.bin", buffer.host_data, shape.area() * 2);



	image_data.workspace.pull();
	writeBinary("Gradient.bin", image_data.workspace.host_data, 2*shape.area() );
	image_data.workspace.destroyHost();

	buffer.destroyDevice();

}

void prepareReferencePC(ReferenceContextPC<Npp8u>& ref_context) {
	ref_context.workspace.createDevice();

	Npp8u* image = ref_context.image.device_data;
	cufftComplex* workspace = ref_context.workspace.device_data;
	cv::Size shape = ref_context.image_shape;

	ref_context.image.pull();
	writeBinary("ref_image.bin", ref_context.image.host_data, shape.area());
	
	//Compute Bounding Box for Gradient Calculation
	double quad[4][2];
	double coeff[2][3] = { {1, 0, 0}, {0, 1, 0}};
	nppiGetAffineQuad(ref_context.box, quad, coeff);

	DualData<double> d_quad = DualData<double>("Reference Quadrangle Bounds", 8);

	d_quad.createHost();
	for (int i = 0; i < 8; ++i) {
		d_quad.host_data[i] = quad[i/2][i%2];
	}
	d_quad.push();
	
	ref_context.area = area(d_quad.host_data);
	//std::cout << "Ref Area: " << area(d_quad.host_data) << std::endl;
	//print("Ref Quad", d_quad.host_data, 8);

	//Create buffer for gradient information
	DualData<float> buffer = DualData<float>("Reference Gradient Buffer", 2 * shape.area());
	buffer.createDevice();

	//Compute Image Gradient
	gradient(image, d_quad.device_data, shape, buffer);
	float* gx = buffer.device_data;
	float* gy = buffer.device_data + shape.area();

	//Repack Gradient into FFT workspace
	int gridsize = 0;
	int blocksize = 0;
	CUDA_CHECK( cudaOccupancyMaxPotentialBlockSize(&gridsize, &blocksize, repack_pc, 0) );
	repack_pc<<<gridsize, blocksize>>>(workspace, gx, gy, shape.area());
	
	buffer.destroyDevice();

	//ref_context.workspace.pull();
	//writeBinary("R_Gradient.bin", output.host_data, shape.area());
	//output.destroyHost();

	//Clean up bounding box
	d_quad.destroyDevice();
	d_quad.destroyHost();

	//Take FFT
	size_t w = 0;
	cufftHandle plan;
	cufftResult r;

	r = cufftCreate(&plan);
	assert(r == CUFFT_SUCCESS);

	r = cufftMakePlan2d(plan, shape.width, shape.height, CUFFT_C2C, &w);
	assert(r == CUFFT_SUCCESS);

	r = cufftXtExec(plan, workspace, workspace, CUFFT_FORWARD);
	assert(r == CUFFT_SUCCESS);

	cufftDestroy(plan);
}

void prepareReferenceNGF(ReferenceContextNGF<Npp8u>& ref_context) {
	ref_context.dx_workspace.createDevice();
	ref_context.dy_workspace.createDevice();
	ref_context.mix_workspace.createDevice();

	Npp8u* image = ref_context.image.device_data;
	cv::Size shape = ref_context.image_shape;

	ref_context.image.pull();
	writeBinary("ref_image.bin", ref_context.image.host_data, shape.area());

	cufftComplex* dx_buffer_out = ref_context.dx_workspace.device_data;
	cufftComplex* dy_buffer_out = ref_context.dy_workspace.device_data;
	cufftComplex* mix_buffer_out = ref_context.mix_workspace.device_data;

	//float* dx_buffer = reinterpret_cast<float*>(ref_context.dx_workspace.device_data);
	//float* dy_buffer = reinterpret_cast<float*>(ref_context.dy_workspace.device_data);
	//float* mix_buffer = reinterpret_cast<float*>(ref_context.mix_workspace.device_data);

	//Compute Bounding Box
	double quad[4][2];
	double coeff[2][3] = { {1, 0, 0}, {0, 1, 0}};
	nppiGetAffineQuad(ref_context.box, quad, coeff);

	DualData<double> d_quad = DualData<double>("Reference Quadrangle Bounds", 8);

	d_quad.createHost();
	for (int i = 0; i < 8; ++i) {
		d_quad.host_data[i] = quad[i/2][i%2];
	}
	d_quad.push();


	ref_context.area = area(d_quad.host_data);

	//Create buffer for gradient information
	DualData<float> buffer = DualData<float>("Reference Gradient Buffer", 2 * shape.area());
	buffer.createDevice();

	//Compute Image Gradient
	gradient(image, d_quad.device_data, shape, buffer);
	float* gx = buffer.device_data;
	float* gy = buffer.device_data + shape.area();

	//Repack Gradient into FFT workspace
	int gridsize = 0;
	int blocksize = 0;
	CUDA_CHECK( cudaOccupancyMaxPotentialBlockSize(&gridsize, &blocksize, repack_ngf, 0) );
	repack_ngf<<<gridsize, blocksize>>>(dx_buffer_out, dy_buffer_out, mix_buffer_out, gx, gy, shape.area());

	buffer.destroyDevice();

	ref_context.dx_workspace.pull();
	writeBinary("rdx.bin", ref_context.dx_workspace.host_data, shape.area());

	ref_context.dy_workspace.pull();
	writeBinary("rdy.bin", ref_context.dy_workspace.host_data, shape.area());

	ref_context.mix_workspace.pull();
	writeBinary("rmix.bin", ref_context.mix_workspace.host_data, shape.area());

	//output.destroyHost();

	//Clean up bounding box
	d_quad.destroyDevice();
	d_quad.destroyHost();

	//Take FFT
	size_t w = 0;
	cufftHandle plan;
	cufftResult r;

	r = cufftCreate(&plan);
	assert(r == CUFFT_SUCCESS);

	r = cufftMakePlan2d(plan, shape.width, shape.height, CUFFT_C2C, &w);
	assert(r == CUFFT_SUCCESS);

	r = cufftXtExec(plan, dx_buffer_out, dx_buffer_out, CUFFT_FORWARD);
	assert(r == CUFFT_SUCCESS);

	r = cufftXtExec(plan, dy_buffer_out, dy_buffer_out, CUFFT_FORWARD);
	assert(r == CUFFT_SUCCESS);

	r = cufftXtExec(plan, mix_buffer_out, mix_buffer_out, CUFFT_FORWARD);
	assert(r == CUFFT_SUCCESS);

	ref_context.dx_workspace.pull();
	writeBinary("frdx.bin", ref_context.dx_workspace.host_data, shape.area());

	ref_context.dy_workspace.pull();
	writeBinary("frdy.bin", ref_context.dy_workspace.host_data, shape.area());

	ref_context.mix_workspace.pull();
	writeBinary("frmix.bin", ref_context.mix_workspace.host_data, shape.area());

	cufftDestroy(plan);
}



void prepareReferencePC(DualData<Npp8u> &input, NppiRect box, DualData<cufftComplex> &output, cv::Size& shape) {
	output.createDevice();

	Npp8u* image = input.device_data;
	cufftComplex* out = output.device_data;
	
	DualData<float> buffer = DualData<float>("Reference Gradient Buffer", 2 * shape.area());
	buffer.createDevice();

	int gridsize = 0, blocksize = 0;
	CUDA_CHECK( cudaOccupancyMaxPotentialBlockSize(&gridsize, &blocksize, repack_pc, 0) );

	//Compute Bounding Box
	double quad[4][2];
	double coeff[2][3] = { {1, 0, 0}, {0, 1, 0}};
	nppiGetAffineQuad(box, quad, coeff);

	DualData<double> d_quad = DualData<double>("Reference Quadrangle Bounds", 8);

	d_quad.createHost();
	for (int i = 0; i < 8; ++i) {
		d_quad.host_data[i] = quad[i/2][i%2];
	}
	d_quad.push();

	//Take Gradient
	gradient(image, d_quad.device_data, shape, buffer);
	float* gx = buffer.device_data;
	float* gy = buffer.device_data + shape.area();
	repack_pc<<<gridsize, blocksize>>>(out, gx, gy, shape.area());

	output.pull();
	writeBinary( "R_Gradient.bin", output.host_data, shape.area() );
	output.destroyHost();
	
	d_quad.destroyDevice();
	d_quad.destroyHost();


	//Take FFT
	size_t w = 0;
	cufftHandle plan;
	cufftResult r;
	//output.pull();
	//writeBinary("RGTest.bin", output.host_data, shape.area());

	r = cufftCreate(&plan);
	assert(r == CUFFT_SUCCESS);

	r = cufftMakePlan2d(plan, shape.width, shape.height, CUFFT_C2C, &w);
	assert(r == CUFFT_SUCCESS);
	
	r = cufftXtExec(plan, out, out, CUFFT_FORWARD);
	assert(r == CUFFT_SUCCESS);

	cufftDestroy(plan);

	buffer.destroyDevice();
}


void populateTransform(SearchContext &sweep, WarpContext<Npp8u> &image_data) {

	unsigned int nBatch = image_data.nBatch;
	Parameter theta = sweep.theta;
	Parameter sw = sweep.scale_w;
	Parameter sh = sweep.scale_h;
	cv::Size shape = image_data.image_shape;

	image_data.image.push();
	image_data.batch.createDevice();
	image_data.transformBuffer.createHost();
	
	const double c_w = double(shape.width) / 2;
	const double c_h = double(shape.height) / 2;

	size_t l = 0;

	for (size_t i = theta.initial; i < theta.N; ++i) {
		for (size_t j = sw.initial; j < sw.N; ++j) {
			for (size_t k = sh.initial; k < sh.N; ++k) {

				if (l >= nBatch) {
					break;
				}
				std::cout << "theta.delta " << theta.delta << std::endl;
				Npp64f angle = theta.start + theta.delta * i;
				Npp64f scale_w = sw.start + sw.delta * j;
				Npp64f scale_h = sh.start + sh.delta * k;
			
				//TODO: This needs to be changed if multithreading. 
				printf("%i, Queuing parameters (%f, %f, %f) \n", l, angle, scale_w, scale_h);

				double c = cos(angle);
				double s = sin(angle);
				
				const size_t idx = static_cast<size_t>(6) * l;
				image_data.transformBuffer.host_data[idx + 0] = scale_w * c;
				image_data.transformBuffer.host_data[idx + 1] = scale_h * s;
				image_data.transformBuffer.host_data[idx + 2] = -c * scale_w * c_w - scale_h * s * c_h + c_w;

				image_data.transformBuffer.host_data[idx + 3] = -scale_w * s;
				image_data.transformBuffer.host_data[idx + 4] = scale_h * c;
				image_data.transformBuffer.host_data[idx + 5] = -scale_h * c * c_h + scale_w * s * c_w + c_h;
				l++;
			}
			sh.initial = 0;
			if (l >= nBatch) {
				break;
			}
		}
		sw.initial = 0;
		if (l >= nBatch) {
			break;
		}
	}
	theta.initial = 0;

	image_data.transformBuffer.push();
	//image_data.transformBuffer.destroyHost();

	//image_data.transformBuffer.pull();
	//print("transformData", image_data.transformBuffer.host_data, 12);

	image_data.batchList.createHost();
	
	for (size_t l = 0; l < nBatch; ++l) {
		size_t batch_increment = static_cast<size_t>(l) * static_cast<size_t>(shape.area());

		NppiWarpAffineBatchCXR Job;
		Job.pSrc = (void*)(image_data.image.device_data);
		Job.pDst = (void*)(image_data.batch.device_data +batch_increment);
		Job.nSrcStep = sizeof(Npp8u) * shape.width;
		Job.nDstStep = sizeof(Npp8u) * shape.width;
		Job.pCoeffs = &image_data.transformBuffer.device_data[6 * l];
		if (image_data.batchList.host_data != NULL) {
			image_data.batchList.host_data[l] = Job;
		}
		else {
			throw std::runtime_error(Formatter() << "Near SegFault: Host BatchList is NULL");
		}
	}

	image_data.batchList.push();

	NPP_CHECK( nppiWarpAffineBatchInit(image_data.batchList.device_data, nBatch) );

	image_data.bboxBuffer.createHost();
	auto transform = image_data.transformBuffer.host_data;
	auto box = image_data.bboxBuffer.host_data;
	auto src_box = image_data.box;

	for (int l = 0; l < nBatch; ++l) {
		const size_t t_idx = static_cast<size_t>(6) * l;
		const size_t b_idx = static_cast<size_t>(8) * l;
		double temp_box[4][2];
		double temp_transform[2][3] = { { transform[t_idx + 0], transform[t_idx + 1], transform[t_idx + 2] }, { transform[t_idx + 3], transform[t_idx+4], transform[t_idx+5] } };
		print(src_box);

		NPP_CHECK( nppiGetAffineQuad(src_box, temp_box, temp_transform) );

		int c = 0;
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 2; ++j) {
				box[b_idx + c] = temp_box[i][j];
				c = c + 1;
			}
		}
		auto quad = &box[b_idx];
		std::cout << "Area: " << area(quad) << std::endl;

	}

	print(box, 8*4);
	std::cout << "area: " << area(box) << std::endl;
	image_data.bboxBuffer.push();

	//NppiRect bbox = { (int)0, (int)0, (int)shape.width, (int)shape.height };
	//NppiSize nppi_shape = { (int) shape.width, (int) shape.height };
	//std::cout << std::endl << "shape: " << shape.width << ":" << shape.height << std::endl;
	//NPP_CHECK( nppiWarpAffineBatch_8u_C1R(nppi_shape, bbox, bbox, NPPI_INTER_LINEAR, image_data.batchList.device_data, nBatch) );


}

struct Location {
	int x;
	int y;
};

Location convertXY(int loc, cv::Size &shape) {
	//std::cout << "Interior Location: " << loc << std::endl;
	Location l;
	l.y = (int)(loc / shape.height);
	l.x = loc - (l.y * shape.height);

	//l.y = (int)(loc / shape.height);
	//l.x = loc - (l.y * shape.height);

	std::cout << "before center: " << l.y << ", "<< l.x << std::endl;

	if (l.y > shape.width / 2) {
		l.y -= shape.width;

	}
	if (l.x> shape.height / 2) {
		l.x -= shape.height;
	}

	std::cout << "location: " << l.y << ", "<< l.x << std::endl;
	

	return l;
}

void print(Location l) {
	std::cout << "(" << l.x << "," << l.y << ")" << std::endl;
}

std::vector<std::pair<size_t, cufftComplex>> computePC(ReferenceContextPC<Npp8u>& ref_context, WarpContextPC<Npp8u>& warp_context, const unsigned int k) {

	const unsigned int nBatch = warp_context.nBatch;

	cv::Size shape = warp_context.image_shape;
	cv::Size batch_shape(shape.width, shape.height * nBatch);

	cufftComplex* batch_data = warp_context.workspace.device_data;
	cufftComplex* ref_data = ref_context.workspace.device_data;

	//Setup FFT

	cufftHandle plan;
	cufftResult r;

	r = cufftCreate(&plan);
	assert(r == CUFFT_SUCCESS);

	size_t w = 0;
	long long* n = (long long*)malloc(sizeof(long long) * 2);
	n[0] = shape.width;
	n[1] = shape.height;

	r = cufftXtMakePlanMany(plan, 2, n, NULL, 1, 0, CUDA_C_32F, NULL, 1, 0, CUDA_C_32F, (long long)nBatch, &w, CUDA_C_32F);
	assert(r == CUFFT_SUCCESS);

	//Forward FFT
	r = cufftXtExec(plan, batch_data, batch_data, CUFFT_FORWARD);
	assert(r == CUFFT_SUCCESS);

	//Compute similarity metric
	int gridsize = 0, blocksize = 0;
	CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&gridsize, &blocksize, pc_kernel, 0));
	pc_kernel<<<gridsize, blocksize >> > (batch_data, ref_data, nBatch * shape.area(), shape.area());

	warp_context.workspace.pull();
	writeBinary("pc_surface.bin", warp_context.workspace.host_data, shape.area());

	//Reverse FFT
	r = cufftXtExec(plan, batch_data, batch_data, CUFFT_INVERSE);
	assert(r == CUFFT_SUCCESS);

	//Get the top k peaks in this match
	auto location_list = kselect_max(batch_data, batch_shape.area(), k);

	free(n);
	cufftDestroy(plan);

	return location_list;
}

std::vector<std::pair<size_t, cufftComplex>> computeNGF(ReferenceContextNGF<Npp8u>& ref_context, WarpContextNGF<Npp8u>& warp_context, const unsigned int k) {

	const unsigned int nBatch = warp_context.nBatch;

	cv::Size shape = warp_context.image_shape;
	cv::Size batch_shape(shape.width, shape.height * nBatch);

	cufftComplex* mov_dx_data_out = warp_context.workspace_dx.device_data;
	cufftComplex* mov_dy_data_out = warp_context.workspace_dy.device_data;
	cufftComplex* mov_mix_data_out = warp_context.workspace_mix.device_data;

	//float* mov_dx_data = reinterpret_cast<float*>(warp_context.workspace_dx.device_data);
	//float* mov_dy_data = reinterpret_cast<float*>(warp_context.workspace_dy.device_data);
	//float* mov_mix_data = reinterpret_cast<float*>(warp_context.workspace_mix.device_data);

	//float* ref_dx_data = reinterpret_cast<float*>(ref_context.dx_workspace.device_data);
	//float* ref_dy_data = reinterpret_cast<float*>(ref_context.dy_workspace.device_data);
	//float* ref_mix_data = reinterpret_cast<float*>(ref_context.mix_workspace.device_data);

	cufftComplex* ref_dx_data =  ref_context.dx_workspace.device_data;
	cufftComplex* ref_dy_data =  ref_context.dy_workspace.device_data;
	cufftComplex* ref_mix_data = ref_context.mix_workspace.device_data;

	double* bbox_list = warp_context.bboxBuffer.device_data;

	//Setup FFT

	cufftHandle plan;
	cufftResult r;

	r = cufftCreate(&plan);
	assert(r == CUFFT_SUCCESS);

	size_t w = 0;
	long long* n = (long long*)malloc(sizeof(long long) * 2);
	n[0] = shape.width;
	n[1] = shape.height;

	r = cufftXtMakePlanMany(plan, 2, n, NULL, 1, 0, CUDA_C_32F, NULL, 1, 0, CUDA_C_32F, (long long)nBatch, &w, CUDA_C_32F);
	assert(r == CUFFT_SUCCESS);

	//Forward FFT : DX^2
	r = cufftXtExec(plan, mov_dx_data_out, mov_dx_data_out, CUFFT_FORWARD);
	assert(r == CUFFT_SUCCESS);

	//Forward FFT: DY^2
	r = cufftXtExec(plan, mov_dy_data_out, mov_dy_data_out, CUFFT_FORWARD);
	assert(r == CUFFT_SUCCESS);

	//Forward FFT: Mix
	r = cufftXtExec(plan, mov_mix_data_out, mov_mix_data_out, CUFFT_FORWARD);
	assert(r == CUFFT_SUCCESS);

	warp_context.workspace_dx.pull();
	writeBinary("fgx.bin", warp_context.workspace_dx.host_data, shape.area());

	warp_context.workspace_dy.pull();
	writeBinary("fgy.bin", warp_context.workspace_dy.host_data, shape.area());

	warp_context.workspace_mix.pull();
	writeBinary("fmix.bin", warp_context.workspace_mix.host_data, shape.area());
	
	//Compute similarity metric
	int gridsize = 0, blocksize = 0;
	CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&gridsize, &blocksize, ngf_kernel, 0));

	ngf_kernel<<<gridsize, blocksize>>>(mov_dx_data_out, mov_dy_data_out, mov_mix_data_out, ref_dx_data, ref_dy_data, ref_mix_data, nBatch * shape.area(), shape.area());


	warp_context.workspace_dx.pull();
	writeBinary("fcom.bin", warp_context.workspace_dx.host_data, shape.area());

	//Reverse FFT

	//r = cufftXtMakePlanMany(plan, 2, n, NULL, 1, 0, CUDA_C_32F, NULL, 1, 0, CUDA_R_32F, (long long)nBatch, &w, CUDA_C_32F);
	//assert(r == CUFFT_SUCCESS);

	r = cufftXtExec(plan, mov_dx_data_out, mov_dx_data_out, CUFFT_INVERSE);
	assert(r == CUFFT_SUCCESS);

	normalize<<<gridsize, blocksize>>>(mov_dx_data_out, bbox_list, ref_context.area, nBatch*shape.area(), shape.area());

	warp_context.workspace_dx.pull();
	writeBinary("ngf_surface.bin", warp_context.workspace_dx.host_data, shape.area());


	//Get the top k peaks in this match
	auto location_list = kselect_max(mov_dx_data_out, batch_shape.area(), k);
	
	free(n);
	cufftDestroy(plan);

	return location_list;
}


std::vector<std::pair<size_t, cufftComplex>> computePhaseCorrelationMultiple(DualData<cufftComplex>& reference, DualData<cufftComplex>& batchBuffer, const unsigned int nBatch, cv::Size& shape, const unsigned int k) {
	cufftHandle plan;
	cufftResult r;

	r = cufftCreate(&plan);
	assert(r == CUFFT_SUCCESS);

	cv::Size batch_shape(shape.width, shape.height * nBatch);
	cufftComplex* batch_data = batchBuffer.device_data;
	cufftComplex* ref_data = reference.device_data;

	size_t w = 0;
	long long* n = (long long*)malloc(sizeof(long long) * 2);
	n[0] = shape.width;
	n[1] = shape.height;

	r = cufftXtMakePlanMany(plan, 2, n, NULL, 1, 0, CUDA_C_32F, NULL, 1, 0, CUDA_C_32F, (long long)nBatch, &w, CUDA_C_32F);
	assert(r == CUFFT_SUCCESS);

	r = cufftXtExec(plan, batch_data, batch_data, CUFFT_FORWARD);
	assert(r == CUFFT_SUCCESS);

	int gridsize = 0, blocksize = 0;
	CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&gridsize, &blocksize, pc_kernel, 0));

	pc_kernel<<<gridsize, blocksize >> > (batch_data, ref_data, nBatch * shape.area(), shape.area());

	r = cufftXtExec(plan, batch_data, batch_data, CUFFT_INVERSE);
	assert(r == CUFFT_SUCCESS);

	auto location_list = kselect_max(batch_data, batch_shape.area(), k);

	free(n);
	cufftDestroy(plan);

	return location_list;
}

std::pair<size_t, cufftComplex> computePhaseCorrelation(DualData<cufftComplex>& reference, DualData<cufftComplex>& batchBuffer, const unsigned int nBatch, cv::Size& shape) {
	cufftHandle plan;
	cufftResult r;

	r = cufftCreate(&plan);
	assert(r == CUFFT_SUCCESS);

	cv::Size batch_shape(shape.width, shape.height * nBatch);
	cufftComplex* batch_data = batchBuffer.device_data;
	cufftComplex* ref_data = reference.device_data;

	size_t w = 0;
	long long *n = (long long*)malloc(sizeof(long long) * 2);
	n[0] = shape.width;
	n[1] = shape.height;

	r = cufftXtMakePlanMany(plan, 2, n, NULL, 1, 0, CUDA_C_32F, NULL, 1, 0, CUDA_C_32F, (long long) nBatch, &w, CUDA_C_32F);
	assert(r == CUFFT_SUCCESS);

	r = cufftXtExec(plan, batch_data, batch_data, CUFFT_FORWARD);
	assert(r == CUFFT_SUCCESS);

	//batchBuffer.pull();
	//writeBinary("FFTTest.bin", batchBuffer.host_data, shape.area());

	int gridsize = 0, blocksize = 0;
	CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&gridsize, &blocksize, pc_kernel, 0));

	pc_kernel<<<gridsize, blocksize>>>(batch_data, ref_data, nBatch * shape.area(), shape.area());

	r = cufftXtExec(plan, batch_data, batch_data, CUFFT_INVERSE);
	assert(r == CUFFT_SUCCESS);

	//batchBuffer.pull();
	//writeBinary("PCTest.bin", batchBuffer.host_data, shape.area());
	//auto testi = toMat(batchBuffer.host_data, batch_shape);
	//print("host_data", buffer.host_data, dsize.area());
	//showImage(testi, "Correlation");

	auto location_pair = argmax(batch_data, batch_shape.area());
	free(n);
	cufftDestroy(plan);
	return location_pair;
}


void warp(WarpContext<Npp8u> &image_data) {
	unsigned int nBatch = image_data.nBatch;
	cv::Size shape = image_data.image_shape;

	NppiRect bbox = { (int)0, (int)0, (int)shape.width, (int)shape.height };
	NppiSize nppi_shape = { (int) shape.width, (int) shape.height };
	std::cout << std::endl << "shape: " << shape.width << ":" << shape.height << std::endl;
	NPP_CHECK( nppiWarpAffineBatch_8u_C1R(nppi_shape, bbox, bbox, NPPI_INTER_LINEAR, image_data.batchList.device_data, nBatch) );
}

void warp1(WarpContextPC<Npp8u>& image_data) {
	cv::Size shape = image_data.image_shape;
	NppiRect bbox = { (int)0, (int)0, (int)shape.width, (int)shape.height };
	NppiSize nppi_shape = { shape.width, shape.height };

	double c_w = double(shape.width) / 2;
	double c_h = double(shape.height) / 2;

	double angle = 3.1415/2;
	double c = cos(angle);
	double s = sin(angle);

	const double aCoeff[2][3] = {
								{c, s, -c*c_w - s*c_h + c_w},
								{-s, c, -c*c_h + s*c_w + c_h}
	};

	int step = shape.width * sizeof(Npp8u);

	NPP_CHECK( nppiWarpAffine_8u_C1R(image_data.image.device_data, nppi_shape, step, bbox,
						  image_data.batch.device_data, step, bbox, aCoeff, NPPI_INTER_LINEAR) );
}

void adjustSweep(size_t index, SearchContext& sweep) {
	unsigned int angle_id = (int)(index / (sweep.scale_h.N * sweep.scale_w.N));
	unsigned int remain = index % (sweep.scale_h.N * sweep.scale_w.N);
	unsigned int sw_id = (int)(remain / sweep.scale_h.N);
	unsigned int sh_id = (int)(remain % sweep.scale_h.N);

	sweep.theta.initial = angle_id;
	sweep.scale_h.initial = sh_id;
	sweep.scale_w.initial = sw_id;

	printf("Starting search at (%i, %i, %i) \n", angle_id, sw_id, sh_id);
}

std::vector<Result> interpretLocations(std::vector<std::pair<size_t, cufftComplex>> gloc_list, SearchContext& sweep, WarpContext<Npp8u>& context, const unsigned int k) {
	unsigned int nBatch = context.nBatch;
	cv::Size shape = context.image_shape;

	std::vector<Result> topK(k);

	for (size_t i = 0; i < k; ++i) {
		std::pair<size_t, cufftComplex> gloc = gloc_list[i];
		size_t loc = (int)(gloc.first / shape.area());
		assert(loc < sweep.theta.N* sweep.scale_w.N* sweep.scale_h.N);
		loc = loc + (sweep.scale_h.N * sweep.scale_w.N) * (sweep.theta.initial) + (sweep.scale_h.N) * (sweep.scale_w.initial) + (sweep.scale_h.initial);

		unsigned int angle_id = (int)(loc / (sweep.scale_h.N * sweep.scale_w.N));
		unsigned int remain = loc % (sweep.scale_h.N * sweep.scale_w.N);
		unsigned int sw_id = (int)(remain / sweep.scale_h.N);
		unsigned int sh_id = (int)(remain % sweep.scale_h.N);

		//Get search result parameters
		Npp64f theta_result = (angle_id)*sweep.theta.delta + sweep.theta.start;
		Npp64f scale_w_result = (sw_id)*sweep.scale_w.delta + sweep.scale_w.start;
		Npp64f scale_h_result = (sh_id)*sweep.scale_h.delta + sweep.scale_h.start;
		Npp64f value = mag(gloc.second);

		//Get shift (translation) parameters. (For debugging)
		size_t space = (int)(gloc.first % shape.area());

		std::cout << "space: " << space << std::endl;

		auto shift = convertXY(space, shape);
		Npp64f dh = (Npp64f) shift.x;
		Npp64f dw = (Npp64f) shift.y;

		//std::cout << "-----" << std::endl;
		//std::cout << theta_result <<" " << scale_h_result << " " << scale_w_result << " " << value << " " << dh << " " << dw <<std::endl;
		//std::cout << "-----" << std::endl;
		
		topK[i] = Result(theta_result, scale_h_result, scale_w_result, value, dh, dw);

	}

	return topK;
}

void interpretLocation(std::pair<size_t, cufftComplex> gloc, SearchContext &sweep, WarpContext<Npp8u> &context) {
	unsigned int nBatch = context.nBatch;
	cv::Size shape = context.image_shape;
	size_t loc = (int)(gloc.first / shape.area());
	std::cout << "Location: " << loc << std::endl;
	assert(loc < sweep.theta.N * sweep.scale_w.N * sweep.scale_h.N);
	loc = loc + (sweep.scale_h.N * sweep.scale_w.N) * (sweep.theta.initial) + (sweep.scale_h.N) * (sweep.scale_w.initial) + (sweep.scale_h.initial);
	std::cout << "Location (after shift): " << loc << std::endl;
	if ( mag(gloc.second) > sweep.value) {
		unsigned int angle_id = (int)(loc / (sweep.scale_h.N * sweep.scale_w.N));
		unsigned int remain = loc % (sweep.scale_h.N * sweep.scale_w.N);
		unsigned int sw_id = (int)(remain / sweep.scale_h.N);
		unsigned int sh_id = (int)(remain % sweep.scale_h.N);

		sweep.theta.result = (angle_id) * sweep.theta.delta + sweep.theta.start;
		sweep.scale_w.result = (sw_id) * sweep.scale_w.delta + sweep.scale_w.start;
		sweep.scale_h.result = (sh_id) * sweep.scale_h.delta + sweep.scale_h.start;

		sweep.value = mag(gloc.second);

		//Shift
		size_t space = (int)(gloc.first % shape.area());
		space = (int)(gloc.first - loc * shape.area());
		print(convertXY(space, shape));
	}

}

void printResult(SearchContext& sweep) {
	printf("The Result of the search is (%f, %f, %f) \n", sweep.theta.result, sweep.scale_w.result, sweep.scale_h.result);
}

void searchPC(ReferenceContextPC<Npp8u>& ref_context, SearchContext& sweep, WarpContextPC<Npp8u>& warp_context) {
	cv::Size shape = warp_context.image_shape;

	warp_context.image.pull();
	writeBinary("mov_image.bin", warp_context.image.host_data, shape.area());

	size_t permutations = sweep.theta.N * sweep.scale_w.N * sweep.scale_h.N;
	size_t batch_size = warp_context.nBatch;
	size_t k = sweep.k;

	size_t iters = (size_t)ceil((double)permutations / (double)batch_size);

	size_t starting_idx = 0;

	warp_context.workspace.createDevice();

	for (size_t l = 0; l < iters; ++l) {
		starting_idx = l * batch_size;

		adjustSweep(starting_idx, sweep);

		if ((l == iters - 1) && (permutations % batch_size)) {
			batch_size = permutations % batch_size;
			warp_context.nBatch = batch_size;
		}

		populateTransform(sweep, warp_context);
		warp(warp_context);
		gradient(warp_context);

		auto location_list = computePC(ref_context, warp_context, k);

		auto result_list = interpretLocations(location_list, sweep, warp_context, k);
		merge_k(sweep.topK, result_list, k);

	}

	warp_context.workspace.destroyDevice();
}

void searchNGF(ReferenceContextNGF<Npp8u>& ref_context, SearchContext& sweep, WarpContextNGF<Npp8u>& warp_context) {
	cv::Size shape = warp_context.image_shape;

	warp_context.image.pull();
	writeBinary("mov_image.bin", warp_context.image.host_data, shape.area());

	size_t permutations = sweep.theta.N * sweep.scale_w.N * sweep.scale_h.N;
	size_t batch_size = warp_context.nBatch;
	size_t k = sweep.k;

	size_t iters = (size_t)ceil((double)permutations / (double)batch_size);

	size_t starting_idx = 0;

	warp_context.workspace_dx.createDevice();
	warp_context.workspace_dy.createDevice();
	warp_context.workspace_mix.createDevice();

	for (size_t l = 0; l < iters; ++l) {
		starting_idx = l * batch_size;

		adjustSweep(starting_idx, sweep);

		if ((l == iters - 1) && (permutations % batch_size)) {
			batch_size = permutations % batch_size;
			warp_context.nBatch = batch_size;
		}

		populateTransform(sweep, warp_context);
		warp(warp_context);
		gradient(warp_context);

		auto location_list = computeNGF(ref_context, warp_context, k);

		auto result_list = interpretLocations(location_list, sweep, warp_context, k);
		merge_k(sweep.topK, result_list, k);

	}

	warp_context.workspace_dx.destroyDevice();
	warp_context.workspace_dy.destroyDevice();
	warp_context.workspace_mix.destroyDevice();

}


void correlationSearchMultiple(DualData<cufftComplex>& reference, SearchContext& sweep, WarpContextPC<Npp8u>& batch) {
	cv::Size shape = batch.image_shape;

	size_t permutations = sweep.theta.N * sweep.scale_w.N * sweep.scale_h.N;
	size_t batch_size = batch.nBatch;
	size_t k = sweep.k;
	size_t iters = (size_t)ceil((double)permutations / (double)batch_size);

	size_t starting_idx = 0;

	batch.workspace.createDevice();

	for (size_t l = 0; l < iters; l++) {
		starting_idx = l * batch_size;
		adjustSweep(starting_idx, sweep);

		if ((l == iters - 1) && (permutations % batch_size)) {
			batch_size = permutations % batch_size;
			batch.nBatch = batch_size;
		}

		populateTransform(sweep, batch);
		warp(batch);
		gradient(batch);
		auto location_list = computePhaseCorrelationMultiple(reference, batch.workspace, batch_size, shape, k);
		auto result_list = interpretLocations(location_list, sweep, batch, k);
		merge_k(sweep.topK, result_list, k);

	}


}


void correlationSearch(DualData<cufftComplex>& reference, SearchContext& sweep, WarpContextPC<Npp8u> &batch) {
	cv::Size shape = batch.image_shape;

	size_t permutations = sweep.theta.N * sweep.scale_w.N * sweep.scale_h.N;
	size_t batch_size = batch.nBatch;

	size_t iters = (size_t)ceil((double)permutations / (double)batch_size);

	size_t starting_idx = 0;

	batch.workspace.createDevice();

	for (size_t l = 0; l < iters; l++) {
		starting_idx = l * batch_size;
		adjustSweep(starting_idx, sweep);

		if ((l == iters - 1) && (permutations % batch_size)) {
			batch_size = permutations % batch_size;
			batch.nBatch = batch_size;
		}

		populateTransform(sweep, batch);
		warp(batch);
		gradient(batch);
		auto location_pair = computePhaseCorrelation(reference, batch.workspace, batch_size, shape);
		location_pair.first = location_pair.first;
		interpretLocation(location_pair, sweep, batch);
		printResult(sweep);
		
	}

	batch.workspace.destroyDevice();
}

size_t estimateWarpBatchSize(int* shape, long long batch_size, std::string similarity_type) {
	size_t mem = 0;
	mem = (shape[0] * shape[1] * batch_size) * sizeof(Npp8u);
	return mem;
}

size_t estimateFFTBatchSize(int* shape, long long batch_size, std::string similarity_type) {
	cufftHandle plan;
	cufftResult r;

	size_t fmem = 0; //Memory size internal to fft
	size_t gmem = 0; //Memory size for storage buffers (gradient and fft)
	long long* n = (long long*)malloc(sizeof(long long) * 2);
	n[0] = (long long)shape[0];
	n[1] = (long long)shape[1];

	r = cufftCreate(&plan);
	assert(r == CUFFT_SUCCESS);

	if (similarity_type == "PC") {

		r = cufftXtGetSizeMany(plan, 2, n, NULL, 1, 0, CUDA_C_32F, NULL, 1, 0, CUDA_C_32F, batch_size, &fmem, CUDA_C_32F);
		assert(r == CUFFT_SUCCESS);

		size_t gmem = 0;
		gmem = (shape[0] * shape[1] * sizeof(cufftComplex) * batch_size);
		
		r = cufftDestroy(plan);
		assert(r == CUFFT_SUCCESS);
	}
	else if (similarity_type == "NGF") {
		r = cufftXtGetSizeMany(plan, 2, n, NULL, 1, 0, CUDA_R_32F, NULL, 1, 0, CUDA_C_32F, batch_size, &fmem, CUDA_C_32F);
		assert(r == CUFFT_SUCCESS);

		size_t gemm = 0;
		gmem = 3 * ((size_t)floor(shape[1] / 2) + 1) * shape[0] * sizeof(cufftComplex) * batch_size;

	}

	free(n);
	return fmem + gmem;
}

size_t estimateBatchSize(int* shape, long long max_batch, std::string similarity_type) {


	long long batch_size = max_batch;

	size_t free_mem = 0;
	size_t total_mem = 0;
	CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

	//std::cout << "Free: " << free_mem << " , Total: " << total_mem << std::endl;
	
	double percent = 0.0;
	batch_size = 1;
	while (percent < 0.90) {
		//std::cout << "Computing for batch_size: " << batch_size << std::endl;

		size_t est_mem = 0;
		est_mem += estimateWarpBatchSize(shape, batch_size, similarity_type);
		est_mem += estimateFFTBatchSize(shape, batch_size, similarity_type);
		est_mem += (2 * shape[0] * shape[1] * sizeof(Npp8u));		 //Moving Image
		est_mem += (3 * shape[0] * shape[1] * sizeof(cufftComplex)); //Gradient Workspace
		est_mem += (0 * batch_size * shape[0] * shape[1] * sizeof(size_t)); //Sort Space
		percent = double(est_mem) / free_mem;

		//std::cout << "Estimated working memory: " << est_mem << std::endl;
		//std::cout << "Perc: " << percent << std::endl;

		batch_size += 10;

		if (batch_size > max_batch) {
			batch_size = max_batch;
			break;
		}

	}
	return batch_size;
}


void performPCSearch(unsigned int batch_size, unsigned char* reference, int* rbbox, unsigned char* moving, int* mbbox, int* shape, double* params, double* soln, const unsigned int k) {
	cv::Size cv_shape(shape[0], shape[1]);

	//Setup parameter search space
	//Parameter(start, stop, steps)
	Parameter theta = Parameter(params[0], params[1], params[2]);
	Parameter sw = Parameter(params[3], params[4], params[5]);
	Parameter sh = Parameter(params[6], params[7], params[8]);
	SearchContext sweep = SearchContext(theta, sw, sh, k);

	//Determine Batch Size
	batch_size = (unsigned int)std::min(theta.N * sw.N * sh.N, batch_size);
	batch_size = estimateBatchSize(shape, batch_size, "PC");

	//Determine Buffer Shape
	cv::Size output_shape(cv_shape.width, cv_shape.height * batch_size);

	//Make Dual Datas for Images	
	DualData<Npp8u> referenceDD = DualData<Npp8u>("Reference", (Npp8u*) reference, cv_shape.area(), "HOST");
	DualData<Npp8u> movingDD = DualData<Npp8u>("Moving", (Npp8u*) moving, cv_shape.area(), "HOST");

	//Create bounding boxes (whole image)
	NppiRect ref_box = { rbbox[0], rbbox[1], rbbox[2], rbbox[3] };
	NppiRect mov_box = { mbbox[0], mbbox[1], mbbox[2], mbbox[3] };

	//Move Images to GPU
	movingDD.push();
	referenceDD.push();

	//Storage class for Reference Image
	ReferenceContextPC<Npp8u> ref_context = ReferenceContextPC<Npp8u>(referenceDD, cv_shape);
	ref_context.box = ref_box;

	//Storage class for Moving Image
	WarpContextPC<Npp8u> warp_context = WarpContextPC<Npp8u>(movingDD, cv_shape, batch_size);
	warp_context.box = mov_box;

	//Take gradient and fft of reference image
	prepareReferencePC(ref_context);

	//Perform exhaustive search
	searchPC(ref_context, sweep, warp_context);

	const int soln_width = 6;
	for (size_t i = 0; i < k; ++i) {
			soln[i*soln_width + 0] = sweep.topK[i].value;
			soln[i*soln_width + 1] = sweep.topK[i].angle;
			soln[i*soln_width + 2] = sweep.topK[i].scale_w;
			soln[i*soln_width + 3] = sweep.topK[i].scale_h;
			soln[i*soln_width + 4] = sweep.topK[i].dw;
			soln[i*soln_width + 5] = sweep.topK[i].dh;
	}

	referenceDD.destroyDevice();

	warp_context.batchList.clean();
	warp_context.transformBuffer.clean();
	warp_context.batch.clean();

	warp_context.image.destroyDevice();
}

void performNGFSearch(unsigned int batch_size, unsigned char* reference, int* rbbox, unsigned char* moving, int* mbbox, int* shape, double* params, double* soln, const unsigned int k) {
	cv::Size cv_shape(shape[0], shape[1]);

	//Setup parameter search space
	//Parameter(start, stop, steps)
	Parameter theta = Parameter(params[0], params[1], params[2]);
	Parameter sw = Parameter(params[3], params[4], params[5]);
	Parameter sh = Parameter(params[6], params[7], params[8]);
	SearchContext sweep = SearchContext(theta, sw, sh, k);

	//Determine Batch Size
	batch_size = (unsigned int)std::min(theta.N * sw.N * sh.N, batch_size);
	batch_size = estimateBatchSize(shape, batch_size, "NGF");

	//Determine Buffer Shape
	cv::Size output_shape(cv_shape.width, cv_shape.height * batch_size);

	//Make Dual Datas for Images	
	DualData<Npp8u> referenceDD = DualData<Npp8u>("Reference", (Npp8u*) reference, cv_shape.area(), "HOST");
	DualData<Npp8u> movingDD = DualData<Npp8u>("Moving", (Npp8u*) moving, cv_shape.area(), "HOST");

	//Create bounding boxes (whole image)
	NppiRect ref_box = { rbbox[0], rbbox[1], rbbox[2], rbbox[3] };
	NppiRect mov_box = { mbbox[0], mbbox[1], mbbox[2], mbbox[3] };

	//Move Images to GPU
	movingDD.push();
	referenceDD.push();

	//Storage class for Reference Image
	ReferenceContextNGF<Npp8u> ref_context = ReferenceContextNGF<Npp8u>(referenceDD, cv_shape);
	ref_context.box = ref_box;

//Storage class for Moving Image
	WarpContextNGF<Npp8u> warp_context = WarpContextNGF<Npp8u>(movingDD, cv_shape, batch_size);
	warp_context.box = mov_box;

	//Take gradient and fft of reference image
	prepareReferenceNGF(ref_context);

	//Perform exhaustive search
	searchNGF(ref_context, sweep, warp_context);

	const int soln_width = 6;
	for (size_t i = 0; i < k; ++i) {
			soln[i*soln_width + 0] = sweep.topK[i].value;
			soln[i*soln_width + 1] = sweep.topK[i].angle;
			soln[i*soln_width + 2] = sweep.topK[i].scale_w;
			soln[i*soln_width + 3] = sweep.topK[i].scale_h;
			soln[i*soln_width + 4] = sweep.topK[i].dw;
			soln[i*soln_width + 5] = sweep.topK[i].dh;
	}

	referenceDD.destroyDevice();

	warp_context.batchList.clean();
	warp_context.transformBuffer.clean();
	warp_context.batch.clean();

	warp_context.image.destroyDevice();
}

void performSearchMultiple(unsigned int batch_size, unsigned char* reference, int* rbbox, unsigned char* moving, int * mbbox, int* shape, double* params, double* soln, unsigned int similarity, const unsigned int k) {
	cv::Size cv_shape(shape[0], shape[1]);

	//Setup parameter search space
	//Parameter(start, stop, steps)
	Parameter theta = Parameter(params[0], params[1], params[2]);
	Parameter sw = Parameter(params[3], params[4], params[5]);
	Parameter sh = Parameter(params[6], params[7], params[8]);
	SearchContext sweep = SearchContext(theta, sw, sh, k);

	//Determine Batch Size
	batch_size = (unsigned int)std::min(theta.N * sw.N * sh.N, batch_size);
	batch_size = estimateBatchSize(shape, batch_size, "PC");

	cv::Size output_shape(cv_shape.width, cv_shape.height * batch_size);

	DualData<Npp8u> referenceDD = DualData<Npp8u>("Reference", (Npp8u*) reference, cv_shape.area(), "HOST");
	DualData<Npp8u> movingDD = DualData<Npp8u>("Moving", (Npp8u*) moving, cv_shape.area(), "HOST");

	NppiRect ref_box = { rbbox[0], rbbox[1], rbbox[2], rbbox[3] };
	NppiRect mov_box = { mbbox[0], mbbox[1], mbbox[2], mbbox[3] };

	movingDD.push();
	referenceDD.push();

	DualData<cufftComplex> RFFT = DualData<cufftComplex>("Reference FFT", cv_shape.area());
	prepareReferencePC(referenceDD, ref_box, RFFT, cv_shape);

	WarpContextPC<Npp8u> warp_context = WarpContextPC<Npp8u>(movingDD, cv_shape, batch_size);
	warp_context.box = mov_box;

	correlationSearchMultiple(RFFT, sweep, warp_context);

	const int soln_width = 6;
	for (size_t i = 0; i < k; ++i) {
			soln[i*soln_width + 0] = sweep.topK[i].value;
			soln[i*soln_width + 1] = sweep.topK[i].angle;
			soln[i*soln_width + 2] = sweep.topK[i].scale_w;
			soln[i*soln_width + 3] = sweep.topK[i].scale_h;
			soln[i*soln_width + 4] = sweep.topK[i].dw;
			soln[i*soln_width + 5] = sweep.topK[i].dh;
	}

	referenceDD.destroyDevice();
	RFFT.clean();

	warp_context.batchList.clean();
	warp_context.transformBuffer.clean();
	warp_context.batch.clean();
	warp_context.image.destroyDevice();
}


void performSearchSingle(unsigned int batch_size, unsigned char* reference, int* rbbox, unsigned char* moving, int* mbbox, int* shape, double* params, double* soln, unsigned int similarity) {
	cv::Size cv_shape(shape[0], shape[1]);

	//Parameter(start, stop, steps)
	Parameter theta = Parameter(params[0], params[1], params[2]);
	Parameter sw = Parameter(params[3], params[4], params[5]);
	Parameter sh = Parameter(params[6], params[7], params[8]);

	batch_size = (unsigned int) std::min(theta.N * sw.N * sh.N, batch_size);
	batch_size = estimateBatchSize(shape, batch_size, "PC");

	SearchContext sweep = SearchContext(theta, sw, sh);

	cv::Size output_shape(cv_shape.width, cv_shape.height * batch_size);

	DualData<Npp8u> referenceDD = DualData<Npp8u>("Reference", (Npp8u*) reference, cv_shape.area(), "HOST");
	DualData<Npp8u> movingDD = DualData<Npp8u>("Moving", (Npp8u*) moving, cv_shape.area(), "HOST");

	NppiRect ref_box = { rbbox[0], rbbox[1], rbbox[2], rbbox[3] };
	NppiRect mov_box = { mbbox[0], mbbox[1], mbbox[2], mbbox[3] };

	movingDD.push();
	referenceDD.push();
	
	DualData<cufftComplex> RFFT = DualData<cufftComplex>("Reference FFT", cv_shape.area());
	prepareReferencePC(referenceDD, ref_box, RFFT, cv_shape);

	//reference_spectrum.pull();
	//cv::Mat Ref = toMat(padded_reference.host_data, template_shape);
	//showImage(Ref, "Reference Image");
	//cv::Mat Refft = toMat(reference_spectrum.host_data, template_shape);
	//showImage(Refft, "Reference Spectrum");

	//writeBinary("RTest.bin", padded_reference.host_data, template_shape.area());
	//writeBinary("RFTTest.bin", reference_spectrum.host_data, template_shape.area());

	//cv::Mat Mov = toMat(padded_moving.host_data, template_shape);
	//showImage(Mov, "Moving Image");

	WarpContextPC<Npp8u> warp_context = WarpContextPC<Npp8u>(movingDD, cv_shape, batch_size);
	warp_context.box = mov_box;

	correlationSearch(RFFT, sweep, warp_context);

	soln[0] = sweep.theta.result;
	soln[1] = sweep.scale_w.result;
	soln[2] = sweep.scale_h.result;

	referenceDD.destroyDevice();
	RFFT.clean();

	warp_context.batchList.clean();
	warp_context.transformBuffer.clean();
	warp_context.batch.clean();
	warp_context.image.destroyDevice();
}

int main()
{
	static_assert(std::is_same_v<std::uint8_t, unsigned char>,
    "This library requires std::uint8_t to be implemented as char or unsigned char.");

	static_assert(std::is_same_v<Npp8u, unsigned char>,
    "This library requires Nppu8 to be implemented as char or unsigned char.");

	CUDA_CHECK(cudaSetDevice(0));

	//testArgMax();

	
	const int similarity = 0;
	const int k = 5;
	cv::Mat Moving = readImage("images/rome_image.png");
	//cv::Mat Reference = readImage("images/test_rome_cropped.png");
	cv::Mat Reference = readImage("images/rome_image.png");

	unsigned int resolution = 128;
	cv::Mat ds_Moving = downsampleImage(Moving, resolution);
	cv::Mat ds_Reference = downsampleImage(Reference, resolution);

 	cv::Mat pad_Moving, pad_Reference;
	padImages(ds_Moving, ds_Reference, pad_Moving, pad_Reference);

	showImage(pad_Reference, "Reference Image");
	showImage(pad_Moving, "Moving Image");

	cv::Size cv_shape = pad_Moving.size();
	unsigned int nBatch = 50;
	//int shape[2] = { cv_shape.width, cv_shape.height };
	int shape[2] = { cv_shape.height, cv_shape.width };
	double soln[k * 6] = {0, 0, 0, 0, 0, 0 };
	//double params[9] = {-180, 180, 361, 1, 1, 1, 1, 1, 1};
	double params[9] = { 0, 0, 3, 1, 1, 1, 1, 1, 1 };
	//double params[9] = { -1, 1, 12, 0.8, 1.2, 11, 0.8, 1.2, 10 };

	//print((int*)pad_Reference.data, 3);
	std::cout << "SHAPE " << cv_shape.height << " , " << cv_shape.width << std::endl;
	int rbbox[8] = { 128, 128, 256, 256};// cv_shape.width - 128, cv_shape.height - 128

	//performPCSearch(nBatch, pad_Reference.data, rbbox, pad_Moving.data, rbbox, &shape[0], &params[0], &soln[0], k);
	performNGFSearch(nBatch, pad_Reference.data, rbbox, pad_Moving.data, rbbox, &shape[0], &params[0], &soln[0], k);


	//performSearchMultiple(nBatch, pad_Reference.data, rbbox, pad_Moving.data, rbbox, &shape[0], &params[0], &soln[0], similarity, k);
	//performSearchSingle(nBatch, pad_Reference.data, pad_Moving.data, &shape[0], &params[0], &soln[0], similarity);

	print(soln, 5);
	

 	return 0;

}

