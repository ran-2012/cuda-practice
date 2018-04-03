
//Ran@2018/3/30

#include <exception>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <cstdio>

#include <cuda_runtime.h>

#include "ran_timer.h"
#include "ran_helper_functions.h"

//kernel函数
__global__ void vectorAdd(float *a, float *b, float *c, int num)
{
	//说明： 
	//blockDim：每个block有多少个thread
	//blockIdx：当前block编号
	//threadIdx：当前thread编号
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num)
		return;
	c[i] = a[i] + a[i];
}

int main()
{
	//数据量
	constexpr long num = 500000;
	constexpr long size = num * sizeof(float);

	//计时器
	Timer t;
	
	displayInfo();

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
		//调用所需的block数、每个block中有多少thread
		vectorAdd << <block, threads>> > (ga, gb, gc, num);
		errProc(cudaGetLastError(), "无法启动");

		//取结果
		errProc(cudaMemcpy(rc, gc, size, cudaMemcpyDeviceToHost), "无法读取结果");
		
		//释放显存
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
	
	std::cout << "Press Enter to exit" << std::endl;
	getchar();

	return 0;
}
