#pragma once

//CUFFT
#include<cufft.h>
#include<cufftXt.h>


__device__ __host__ inline cufftComplex operator+(const cufftComplex a, const cufftComplex b) {
	cufftComplex c;
	c.x = float(double(a.x) + double(b.x));
	c.y = float(double(a.y) + double(b.y));
	return c;
}

__device__ __host__ inline cufftComplex operator*(const cufftComplex a, const cufftComplex b) {
	cufftComplex c;
	c.x = float( double(a.x) * double(b.x) - double(a.y) * double(b.y) );
	c.y = float( (double(a.x) * double(b.y)) + (double(a.y) * double(b.x)));
	return c;
}


__device__ __host__ inline cufftComplex operator*(const float a, const cufftComplex b) {
	cufftComplex c;
	c.x = a * b.x;
	c.y = a * b.y;
	return c;
}

__device__ __host__ inline cufftComplex operator/(const cufftComplex a, const float b) {
	cufftComplex c;
	c.x = a.x / b;
	c.y = a.y / b;
	return c;
}

__device__ __host__ inline float mag(const cufftComplex a) {
	float xsq, ysq;
	xsq = a.x * a.x;
	ysq = a.y *a.y;
	return sqrtf(xsq + ysq);
}

__device__ __host__ inline cufftComplex cong(const cufftComplex a) {
	cufftComplex c;
	c.x = a.x;
	c.y = -a.y;
	return c;
}

__device__ __host__ inline cufftComplex zero() {
	cufftComplex c;
	c.x = 0.0;
	c.y = 0.0;
	return c;
}

__device__ __host__ inline bool operator>(const cufftComplex a, const cufftComplex b) {
	return (mag(a) > mag(b));
}

__device__ __host__ inline bool operator<(const cufftComplex a, const cufftComplex b) {
	return (mag(a) < mag(b));
}

struct bigger_real {
	__device__ __host__
	cufftComplex operator()(const cufftComplex a, const cufftComplex b) {
		if (a.x > b.x) return a;
		else return b;
	}
};

struct bigger_mag {
	__device__ __host__
	cufftComplex operator()(const cufftComplex a, const cufftComplex b) {
		if (mag(a) > mag(b)) return a;
		else return b;
	}
};

struct mag_comp {
	__device__ __host__
	bool operator()(const cufftComplex a, const cufftComplex b) {
		return (mag(a) < mag(b));
	}
};

struct comp {
	__device__ __host__
	bool operator()(const float a, const float b) {
		return a < b;
	}
};
