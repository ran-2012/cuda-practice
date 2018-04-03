
//Some useful functions
//Author: Ran
//Time  : 2018/3/31

#pragma once

#include <string>
#include <random>
#include <iostream>
#include <cuda_runtime.h>

//错误处理
void errProc(cudaError_t err, std::string errStr = "")
{
	if (err != cudaSuccess)
	{
		throw std::exception((errStr + cudaGetErrorString(err)).c_str());
	}
}

//生成随机数
template<typename randtype = float> randtype rand()
{
	std::uniform_real_distribution<randtype> uni(0, 10000);
	return uni(std::random_device());
};

//显示一些常用信息
void displayInfo(std::ostream &os = std::cout)
{
	int device;
	cudaDeviceProp prop;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);
	os << "Device name: " << prop.name << std::endl;
	os << "Device memory: " << prop.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
	os << "Memory Frequency: " << prop.memoryClockRate / 1000 << "MHz" << std::endl;
	os << "MultiProcessor: " << prop.multiProcessorCount << std::endl;
	os << "Clock rate: " << prop.clockRate / 1000 << "MHz" << std::endl;
	os << "Max threads pre multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
	os << "Max blocks: x: " << prop.maxGridSize[0]
		<< " y: " << prop.maxGridSize[1]
		<< " z: " << prop.maxGridSize[2] << std::endl;
	os << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
	os << "Max threads in each dims: x: " << prop.maxThreadsDim[0]
		<< " y: " << prop.maxThreadsDim[1]
		<< " z: " << prop.maxThreadsDim[2] << std::endl;
	os << "Warp size:" << prop.warpSize << std::endl;
}