
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

void displayInfo(std::ostream &is = std::cout)
{
	int device;
	cudaDeviceProp prop;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);
	is << "Device name: " << prop.name << std::endl;
	is << "Device memory: " << prop.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
	is << "Memory Frequency: " << prop.memoryClockRate / 1000 << "MHz" << std::endl;
	is << "MultiProcessor: " << prop.multiProcessorCount << std::endl;
	is << "Clock rate: " << prop.clockRate / 1000 << "MHz" << std::endl;
	is << "Max threads pre multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
	is << "Max blocks: x: " << prop.maxGridSize[0]
		<< " y: " << prop.maxGridSize[1]
		<< " z: " << prop.maxGridSize[2] << std::endl;
	is << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
	is << "Max threads in each dims: x: " << prop.maxThreadsDim[0]
		<< " y: " << prop.maxThreadsDim[1]
		<< " z: " << prop.maxThreadsDim[2] << std::endl;
	is << "Warp size:" << prop.warpSize << std::endl;
}