#pragma once


//CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_fp16.h"
#include <cuComplex.h>
#include "helper_cuda.h"

//NPP
#include<npp.h>
#include<nppi_geometry_transforms.h>
#include<nppdefs.h>

#include<string>
#include<assert.h>
#include<iostream>
#include<sstream>
#include<cstdlib>
#include<utility>

class Formatter
{
public:
	Formatter() {}
	~Formatter() {}

	template<typename Type>
	Formatter& operator << (const Type& value) {
		stream_ << value;
		return *this;
	}

	std::string str() const { return stream_.str(); }
	operator std::string() const { return stream_.str(); }

	enum ConvertToString {
		to_str
	};

	std::string operator >> (ConvertToString) { return stream_.str(); }

private:
	std::stringstream stream_;

	Formatter(const Formatter&);
	Formatter& operator = (Formatter&);

};

/*
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
*/
