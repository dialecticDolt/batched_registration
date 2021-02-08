#pragma once

#include<winapifamily.h>

extern "C" __declspec(dllexport) void performPCSearch(unsigned int batch_size, unsigned char* reference, int* rbbox, unsigned char* moving, int* mbbox, int* shape, double* params, double* soln, const unsigned int k);
extern "C" __declspec(dllexport) void performNGFSearch(unsigned int batch_size, unsigned char* reference, int* rbbox, unsigned char* moving, int* mbbox, int* shape, double* params, double* soln, const unsigned int k);

//extern "C" __declspec(dllexport) void performPCSearch(unsigned int batch_size, unsigned char* reference, double* rquad, unsigned char* moving, int* mbbox, int* shape, double* params, double* soln, const unsigned int k);
//extern "C" __declspec(dllexport) void performNGFSearch(unsigned int batch_size, unsigned char* reference, double* rquad, unsigned char* moving, int* mbbox, int* shape, double* params, double* soln, const unsigned int k);

extern "C" __declspec(dllexport) void performSearchSingle(unsigned int batch_size, unsigned char* reference, int* rbbox, unsigned char* moving, int* mbbox, int* shape, double* params, double* soln, unsigned int similarity=0);
extern "C" __declspec(dllexport) void performSearchMultiple(unsigned int batch_size, unsigned char* reference, int* rbbox, unsigned char* moving, int* mbbox, int* shape, double* params, double* soln, unsigned int similarity = 0, const unsigned int k = 5);
extern "C" __declspec(dllexport) int testfunc();
