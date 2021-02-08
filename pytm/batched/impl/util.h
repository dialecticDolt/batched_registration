#pragma once

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

//CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_fp16.h"
#include <cuComplex.h>

//CUFFT
#include<cufft.h>
#include<cufftXt.h>

#include "helper_cuda.h"

#include "singleton.h"
#include "phaseCorrelation.h"
#include "profiler.h"


//Section 1
//Instantiations of Singletons
//Singletons to Manage and Reuse FFT workspace and settings


class fftHandlePC final : public Singleton<fftHandlePC> {
	friend class Singleton<fftHandlePC>;
private:

	fftHandlePC() {};

	fftHandlePC(long long width, long long height, long long nBatch) {
		std::cout << "Creating FFT Handle (Phase Correlation)" << std::endl;
		n = (long long*)malloc(sizeof(long long) * 2);
		n[0] = width;
		n[1] = height;

		cufftResult r = cufftCreate(&plan);
		assert(r == CUFFT_SUCCESS);

		size_t w = 0;
		r = cufftXtMakePlanMany(plan, 2, n, NULL, 1, 0, CUDA_C_32F, NULL, 1, 0, CUDA_C_32F, nBatch, &w, CUDA_C_32F);
		assert(r == CUFFT_SUCCESS);

		handle_status = true;
	};

public:
	cufftHandle plan;
	long long* n;
	bool handle_status = false;

	void create(long long width, long long height, long long nBatch) {
		if (!handle_status) {
			std::cout << "Creating FFT Handle (Phase Correlation)" << std::endl;
			n = (long long*)malloc(sizeof(long long) * 2);
			n[0] = width;
			n[1] = height;

			cufftResult r = cufftCreate(&plan);
			assert(r == CUFFT_SUCCESS);

			size_t w = 0;
			r = cufftXtMakePlanMany(plan, 2, n, NULL, 1, 0, CUDA_C_32F, NULL, 1, 0, CUDA_C_32F, nBatch, &w, CUDA_C_32F);
			assert(r == CUFFT_SUCCESS);
			handle_status = true;
		}
	}

	void destroy() {
		if (handle_status) {
			cufftDestroy(plan);
			free(n);
			handle_status = false;
		}
	}

	~fftHandlePC() {
		if (handle_status) {
			cufftDestroy(plan);
			free(n);
			handle_status = false;
		}
	}

};


//Section 2
//CUDA Error Handling

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA Assert Failed: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define NPP_CHECK(ans) { nppAssert((ans), __FILE__, __LINE__); }
inline void nppAssert(NppStatus code, const char* file, int line, bool abort = true) {
	if (code != NPP_SUCCESS) {
		std::cout << "CODE: " << code << std::endl;
		fprintf(stderr, "CUDA-NPP Assert Failed: %d %s %d\n", _cudaGetErrorEnum(code), file, line);
		if (abort) exit(code);
	}

}

//Section 3
//File IO

void writeBinary(std::string filename, float* data, const unsigned int n) {
	std::fstream file;
	file = std::fstream(filename, std::ios::out | std::ios::binary);
	file.write((char*)data, n * sizeof(float));
	file.close();
}

void writeBinary(std::string filename, cufftComplex* data, const unsigned int n) {
	std::fstream file;
	file = std::fstream(filename, std::ios::out | std::ios::binary);
	file.write((char*)data, n * sizeof(cufftComplex));
	file.close();
}

void writeBinary(std::string filename, Npp8u* data, const unsigned int n) {
	std::fstream file;
	file = std::fstream(filename, std::ios::out | std::ios::binary);
	file.write((char*)data, n * sizeof(Npp8u));
	file.close();
}


//Section 3
//Output

void print(std::string name, Npp8u* v, unsigned int n) {
	std::cout << name << std::endl;
	for (unsigned int i = 0; i < n; ++i) {
		std::cout << v[i] << std::endl;
	}
}


void print(std::string name, float* v, unsigned int n) {
	std::cout << name << std::endl;
	for (unsigned int i = 0; i < n; ++i) {
		std::cout << v[i] << std::endl;
	}
}

void print(std::string name, double* v, unsigned int n) {
	std::cout << name << std::endl;
	for (unsigned int i = 0; i < n; ++i) {
		std::cout << v[i] << std::endl;
	}
}

void print(std::string name, cufftComplex* v, unsigned int n) {
	std::cout << name << std::endl;
	for (unsigned int i = 0; i < n; ++i) {
		std::cout << "(" << v[i].x << "," << v[i].y << ")" << std::endl;
	}
}

void print(int* soln, unsigned int n){
	for (size_t i = 0; i < n; ++i) {
		std::cout << soln[i] << " ";
	}
	std::cout << std::endl;
}

void print(double* soln, unsigned int n){
	for (size_t i = 0; i < n; ++i) {
		std::cout << soln[i] << " ";
	}
	std::cout << std::endl;
}
void print(cufftComplex a) {
	std::cout << "(" << a.x << "," << a.y << "i)" << std::endl;
}

template<typename T>
void print(std::string name, thrust::device_ptr<T> a, const unsigned int n) {
	for (size_t i = 0; i < n; ++i) {
		std::cout << a[i] << " ";
	}
	std::cout << std::endl;
}

void print(NppiRect box) {
	std::cout << "( " << box.x << ", " << box.y << " )" << " : " << "( " << box.height << ", " << box.width << " )" << std::endl;
}

//Section 4
//Conversion

Npp8u* toInt(const cv::Mat& Image) {
	return reinterpret_cast<Npp8u*>(Image.data);
}

cv::Mat toMat(Npp8u* image, const cv::Size& shape) {
	Npp8u* cimage = static_cast<Npp8u*>(malloc(sizeof(Npp8u) * shape.area()));
	std::memcpy(cimage, image, sizeof(Npp8u)* shape.area());
	cv::Mat Image_mat = cv::Mat(shape, CV_8UC1, cimage);
	return Image_mat;
}

cv::Mat toMat(float* image, const cv::Size& shape) {
	float* cimage = static_cast<float*>(malloc(sizeof(float) * shape.area()));
	std::memcpy(cimage, image, sizeof(float) * shape.area());
	cv::Mat Image_mat = cv::Mat(shape, CV_32FC1, cimage);
	cv::Mat Image_out;
	Image_mat.convertTo(Image_out, CV_8UC1);
	return Image_out;
}

cv::Mat toMat(cufftComplex* image, const cv::Size& shape, std::string part = "MAG") {
	float* cimage = static_cast<float*>(malloc(sizeof(float) * shape.area()));

	if (part == "REAL") {
		for (unsigned int i = 0; i < shape.area(); ++i) {
			cimage[i] = image[i].x;
		}
	}
	else if (part == "IMAG") {
		for (unsigned int i = 0; i < shape.area(); ++i) {
			cimage[i] = image[i].x;
		}
	}
	else if (part == "MAG") {
		for (unsigned int i = 0; i < shape.area(); ++i) {
			float r = image[i].x;
			float c = image[i].y;
			cimage[i] = sqrtf((r) * (r)+(c) * (c));
		}
	}

	cv::Mat Image_mat = cv::Mat(shape, CV_32FC1, cimage);

	cv::Mat Image_out;
	Image_mat.convertTo(Image_out, CV_8UC1);
	return Image_out;
}

std::vector<float> linspace(float start, float end, const unsigned int num) {
	std::vector<float> linspace_v;

	if (num == 0) {
		// Do Nothing
	}
	else if (num == 1) {
		linspace_v.push_back(start);
	}
	else {
		double delta = (end - start) / (num - 1);

		for (unsigned int i = 0; i < num - 1; ++i) {
			linspace_v.push_back(start + delta * i);
		}
		linspace_v.push_back(end);
	}

	return linspace_v;
}

unsigned int nextPowerof2(unsigned int n) {
	n--;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 8;
	n |= n >> 16;
	n++;
	return n;
}

cv::Size power2pad(const cv::Size &image_shape) {
	unsigned int width  = unsigned(image_shape.width);
	unsigned int height = unsigned(image_shape.height);
	unsigned int pad_width = nextPowerof2(width);
	unsigned int pad_height = nextPowerof2(height);

	return cv::Size(pad_width, pad_height);
}


cv::Mat readImage(std::string filename) {
	cv::Mat Image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
	if (Image.empty()) {
		throw std::runtime_error(Formatter() << "Failed readImage(): " << filename << " not found. ");
	}
	return Image;
}

cv::Mat padImage(const cv::Mat &Input, const cv::Size &new_shape) {
	cv::Size old_shape = Input.size();
	cv::Scalar color(0, 0, 0);
	unsigned int width_diff = unsigned(new_shape.width - old_shape.width);
	unsigned int height_diff = unsigned(new_shape.height - old_shape.height);

	unsigned int left = (width_diff % 2) ? unsigned(width_diff) / 2 : unsigned(std::floor(float(width_diff) / 2));
	unsigned int right = (width_diff % 2) ? unsigned(width_diff) / 2 : unsigned(std::ceil(float(width_diff) / 2));

	unsigned int top = (height_diff % 2) ? height_diff / 2 : unsigned(std::floor(float(height_diff) / 2));
	unsigned int bot = (height_diff % 2) ? height_diff / 2 : unsigned(std::ceil(float(height_diff) / 2));

	cv::Mat Output;
	cv::copyMakeBorder(Input, Output, top, bot, left+1, right, cv::BORDER_CONSTANT, 0);
	cv::Size test = Output.size();
	return Output;
}

void padImages(const cv::Mat &Moving, const cv::Mat &Reference, cv::Mat &paddedMoving, cv::Mat &paddedReference) {

	cv::Size m_shape = Moving.size();
	cv::Size r_shape = Reference.size();

	unsigned int largest_width = std::max(m_shape.width, r_shape.width);
	unsigned int largest_height = std::max(m_shape.height, r_shape.height);
	
	unsigned int largest = std::max(largest_width, largest_height);

	cv::Size max_shape = cv::Size(largest, largest);
	cv::Size padded_shape = power2pad(max_shape);

	paddedMoving = padImage(Moving, padded_shape);
	paddedReference = padImage(Reference, padded_shape);

}

cv::Mat downsampleImage(const cv::Mat &Input, const unsigned int resolution = 256) {
	auto shape = Input.size();
	float scale = float(1) / std::min(float(shape.width) / resolution, float(shape.height) / resolution);

	unsigned int ds_width =  unsigned(std::ceil(scale * shape.width));
	unsigned int ds_height = unsigned(std::ceil(scale * shape.height));
	cv::Size scaled_shape = cv::Size(ds_width, ds_height);


	cv::Mat Output;
	cv::resize(Input, Output, cv::Size(ds_width, ds_height), 0, 0, cv::INTER_AREA);
	return Output;
}


Npp8u* transferImageCPUtoGPU(const cv::Mat &Image) {
	cv::Size shape = Image.size();
	Npp8u* d_Image;
	size_t imageMemSize = size_t(shape.width) * size_t(shape.height) * sizeof(Npp8u);
	CUDA_CHECK( cudaMalloc((void**)&d_Image, imageMemSize) );
	CUDA_CHECK( cudaMemcpy(d_Image, reinterpret_cast<Npp8u*>(Image.data), imageMemSize, cudaMemcpyHostToDevice) );
	return d_Image;
}

cv::Mat transferImageGPUtoCPU(Npp8u* d_Image, const cv::Size &shape) {
	size_t imageMemSize = size_t(shape.width) * size_t(shape.height) * sizeof(Npp8u);
	Npp8u* h_pImage = static_cast<Npp8u*>( malloc(imageMemSize) );

	CUDA_CHECK( cudaMemcpy((void*) h_pImage, (void*) d_Image, imageMemSize, cudaMemcpyDeviceToHost) );
	cv::Mat h_Image = cv::Mat(shape, CV_8UC1, h_pImage);
	return h_Image;
}

__host__ __device__ 
double area(double* quad) {
	const int n = 4;
	const int dim = 2;
	int j = n - 1;

	double area = 0.0;

	#pragma unroll
	for (int i = 0; i < n; ++i) {
		area += 0.5*(quad[j * dim] + quad[i * dim])* (quad[j * dim + 1] - quad[i * dim + 1]);
		j = i;
	}

	return (area > 0) ? area : -1*area;
}
