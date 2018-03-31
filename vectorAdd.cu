
//Ran@2018/3/30

#include <exception>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <cstdio>

#include <cuda_runtime.h>

#include "timer.h"

//kernel����
__global__ void vectorAdd(float *a, float *b, float *c, int num)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	c[i] = a[i] + a[i];
}

//������
void errProc(cudaError_t err, std::string errStr = "")
{
	if (err != cudaSuccess)
	{
		throw std::exception((errStr + cudaGetErrorString(err)).c_str());
	}
}

int main()
{
	//������
	constexpr long num = 500000;
	constexpr long size = num * sizeof(float);

	//��ʱ��
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
		//�ڴ���
		float *ra=new float[num];
		float *rb=new float[num];
		float *rc=new float[num];

		//�Դ���
		float *ga = NULL;
		float *gb = NULL;
		float *gc = NULL;

		//���������
		auto rand = []()
		{
			std::uniform_real_distribution<float> uni(0, 10000);
			return uni(std::random_device());
		};

		//�����������
		for (int i = 0; i != num; ++i)
		{
			ra[i] = rand();
			rb[i] = rand();
		}

		t.begin();

		//���Դ��з���
		errProc(cudaMalloc(&ga, size), "����Aʧ��");
		errProc(cudaMalloc(&gb, size), "����Bʧ��");
		errProc(cudaMalloc(&gc, size), "����Cʧ��");

		//���Ƶ��Դ�
		errProc(cudaMemcpy(ga, ra, size, cudaMemcpyHostToDevice), "����Aʧ��");
		errProc(cudaMemcpy(gb, rb, size, cudaMemcpyHostToDevice), "����Bʧ��");

		//����

		//ÿ�����̸߳���
		const int threads = 1024;
		//������
		const int block = (num + threads - 1) / threads;
		vectorAdd << <threads, block>> > (ga, gb, gc, num);
		errProc(cudaGetLastError(), "�޷�����");

		//ȡ���
		errProc(cudaMemcpy(rc, gc, size, cudaMemcpyDeviceToHost), "�޷���ȡ���");
		
		//�ͷ��ڴ�
		cudaFree(ga), 
		cudaFree(gb);
		cudaFree(gc);

		t.end();
		std::cout << "GPU������ɣ���ʱ��" << t.time() << std::endl;

		t.reset();
		t.begin();

		for (int i = 0; i < num; ++i)
		{
			rc[i] = ra[i] + rb[i];
		}

		t.end();
		std::cout << "CPU������ɣ���ʱ��" << t.time() << std::endl;

		delete[] ra;
		delete[] rb;
		delete[] rc;
	}
	catch (std::exception e)
	{
		std::cerr << e.what() << std::endl;
	}
	
	std::cout << "��������˳�" << std::endl;
	getchar();

	return 0;
}
