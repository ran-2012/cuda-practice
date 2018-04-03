
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

//kernel����
__global__ void vectorAdd(float *a, float *b, float *c, int num)
{
	//˵���� 
	//blockDim��ÿ��block�ж��ٸ�thread
	//blockIdx����ǰblock���
	//threadIdx����ǰthread���
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num)
		return;
	c[i] = a[i] + a[i];
}

int main()
{
	//������
	constexpr long num = 500000;
	constexpr long size = num * sizeof(float);

	//��ʱ��
	Timer t;
	
	displayInfo();

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
		//���������block����ÿ��block���ж���thread
		vectorAdd << <block, threads>> > (ga, gb, gc, num);
		errProc(cudaGetLastError(), "�޷�����");

		//ȡ���
		errProc(cudaMemcpy(rc, gc, size, cudaMemcpyDeviceToHost), "�޷���ȡ���");
		
		//�ͷ��Դ�
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
	
	std::cout << "Press Enter to exit" << std::endl;
	getchar();

	return 0;
}
