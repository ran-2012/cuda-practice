
//Ran@2018/3/30

#include <exception>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <cstdio>

#include <cuda_runtime.h>

#include "timer.h"

//kernel函数
__global__ void vectorAdd(float *a, float *b, float *c, int num)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	c[i] = a[i] + a[i];
}

//错误处理
void errProc(cudaError_t err, std::string errStr = "")
{
	if (err != cudaSuccess)
	{
		throw std::exception((errStr + cudaGetErrorString(err)).c_str());
	}
}

int main()
{
	//数据量
	constexpr long num = 500000;
	constexpr long size = num * sizeof(float);

	//计时器
	Timer t;
	
	int device;
	cudaDeviceProp prop;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);
	std::cout << "Device name: " << prop.name << std::endl;
	std::cout << "Device memory: " << prop.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
	std::cout << "Memory Frequency: " << prop.memoryClockRate / 1000 << "MHz" << std::endl;
	std::cout << "MultiProcessor: " << prop.multiProcessorCount << std::endl;
	std::cout << "Clock rate: " << prop.clockRate / 1000 << "MHz" << std::endl;
	std::cout << "Max threads pre multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "Max blocks: x: " << prop.maxGridSize[0] 
		<< " y: "<<prop.maxGridSize[1] 
		<< " z: " << prop.maxGridSize[2] << std::endl;
	std::cout << "Max threads per block: " <<  prop.maxThreadsPerBlock << std::endl;
	std::cout << "Max threads in each dims: x: " << prop.maxThreadsDim[0]
		<< " y: " << prop.maxThreadsDim[1]
		<< " z: " << prop.maxThreadsDim[2] << std::endl;
	std::cout << "Warp size:" << prop.warpSize << std::endl;

	try
	{
		//内存中
		float *ra=new float[num];
		float *rb=new float[num];
		float *rc=new float[num];

		//显存中
		float *ga = NULL;
		float *gb = NULL;
		float *gc = NULL;

		//生成随机数
		auto rand = []()
		{
			std::uniform_real_distribution<float> uni(0, 10000);
			return uni(std::random_device());
		};

		//生成随机数据
		for (int i = 0; i != num; ++i)
		{
			ra[i] = rand();
			rb[i] = rand();
		}

		t.begin();

		//在显存中分配
		errProc(cudaMalloc(&ga, size), "分配A失败");
		errProc(cudaMalloc(&gb, size), "分配B失败");
		errProc(cudaMalloc(&gc, size), "分配C失败");

		//复制到显存
		errProc(cudaMemcpy(ga, ra, size, cudaMemcpyHostToDevice), "复制A失败");
		errProc(cudaMemcpy(gb, rb, size, cudaMemcpyHostToDevice), "复制B失败");

		//启动

		//每个块线程个数
		const int threads = 1024;
		//块数量
		const int block = (num + threads - 1) / threads;
		vectorAdd << <threads, block>> > (ga, gb, gc, num);
		errProc(cudaGetLastError(), "无法启动");

		//取结果
		errProc(cudaMemcpy(rc, gc, size, cudaMemcpyDeviceToHost), "无法读取结果");
		
		//释放内存
		cudaFree(ga), 
		cudaFree(gb);
		cudaFree(gc);

		t.end();
		std::cout << "GPU运算完成，用时：" << t.time() << std::endl;

		t.reset();
		t.begin();

		for (int i = 0; i < num; ++i)
		{
			rc[i] = ra[i] + rb[i];
		}

		t.end();
		std::cout << "CPU计算完成，用时：" << t.time() << std::endl;

		delete[] ra;
		delete[] rb;
		delete[] rc;
	}
	catch (std::exception e)
	{
		std::cerr << e.what() << std::endl;
	}
	
	std::cout << "按任意键退出" << std::endl;
	getchar();

	return 0;
}
