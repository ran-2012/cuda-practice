
//
#include <cstdlib>
#include <exception>
#include <cstdio>
#include <utility>
#include <memory>
#include <iostream>
#include <fstream>
#include <cmath>

#include <cuda_runtime.h>

#include "ran_timer.h"
#include "ran_helper_functions.h"

class Matrix;

__global__ void matrixMul(Matrix a, Matrix b, Matrix c);

class Matrix
{
	//�������ڴ�������
	std::unique_ptr<float[]> hostData;
	//�������Դ������
	float *deviceData;
	//��ȣ�ÿ�е�������
	size_t _width;
	//�߶ȣ�����
	size_t _height;
	//������
	size_t _size;

	//��ʼ��CUDA����������
	void initCUDACompution()
	{
		errProc(cudaMalloc(&deviceData, size() * sizeof(float)),
			"could not malloc memory deviceData in device");
		errProc(cudaMemcpy(deviceData, hostData.get(), size() * sizeof(float), cudaMemcpyHostToDevice),
			"could not copy memory from hostData to deviceData");
	}
	void clearDeviceMem()
	{
		if (deviceData != nullptr)
		{
			cudaFree(deviceData);
			deviceData = nullptr;
		}
	}
	//����CUDA����
	void endCUDACompution()
	{
		clearDeviceMem();
	}

public:

	Matrix() :_width(1), _height(1), _size(1), 
		deviceData(nullptr)
	{
		hostData = std::make_unique<float[]>(size());
		zeroing();
	}
	//wΪ��ȣ�hΪ�߶�
	Matrix(size_t w, size_t h) :_width(w), _height(h), _size(w*h), 
		deviceData(nullptr)
	{ 
		hostData = std::make_unique<float[]>(size());
		zeroing();
	}
	//
	Matrix(Matrix& m) :_width(m.width()), _height(m.height()), _size(m.size()),
		deviceData(nullptr)
	{
		hostData = std::make_unique<float[]>(size());
		for (size_t i = 0; i != size(); ++i)
		{
			hostData[i] = m[i];
		}
	}
	~Matrix()
	{
		//delete[] hostData;
	}
	//����Ŀ�ȣ�������
	size_t width() const
	{
		return _width;
	}
	//����ĸ߶ȣ�������
	size_t height() const
	{
		return _height;
	}
	//�����������
	size_t size() const
	{
		return _size;
	}
	//�������������
	void randomize()
	{
		for (size_t i = 0; i != size(); ++i)
		{
			hostData[i] = rand<float>();
		}
	}
	//�������ݹ���
	void zeroing()
	{
		for (size_t i = 0; i != size(); ++i)
		{
			hostData[i] = 0;
		}
	}
	//ͨ��һά�±��������
	float &operator[](size_t id) const
	{
		if (id > size())
			throw std::exception("id out of range in Matrix::operator[]");
		return hostData[id];
	}
	//ͨ����ά�±��������
	float &operator()(size_t x, size_t y) const
	{
		if (x > _width)
			throw std::exception("x out of range in Matrix::operator()");
		if (y > _height)
			throw std::exception("y out of range in Matrix::operator()");
		return hostData[x + y*_width];
	}
	//�������������
	Matrix &operator=(const Matrix& m)
	{
		_height = m.height();
		_width = m.width();
		_size = m.size();

		hostData = std::make_unique<float[]>(size());
		for (size_t i = 0; i != size(); ++i)
		{
			hostData[i] = m[i];
		}
		return *this;
	}
	//��ֵ
	Matrix &operator=(Matrix&& m)
	{
		_height = m.height();
		_width = m.width();
		_size = m.size();

		hostData = std::move(m.hostData);
		return *this;
	}
	//�����Դ��е�����
	__device__ float &accDevice(size_t x, size_t y)
	{
		return deviceData[x + y * _width];
	}
	//����Ŀ��
	__device__ size_t widthDevice() const
	{
		return _width;
	}
	//����ĸ߶�
	__device__ size_t heightDevice() const
	{
		return _width;
	}
	//����˷�
	Matrix operator*(Matrix &m)
	{
		Matrix ret(this->height(), m.width());
		try
		{
			//��״����
			if (width() != m.height())
				throw std::exception("the shape of matrix does not satisfy\
					 the requirement of matrix multiplication in Matrix::operator*()");
			//��ʼ�������Ƶ��Դ�
			initCUDACompution();
			m.initCUDACompution();
			ret.initCUDACompution();

			//ִ��
			matrixMul<<<height(), 32>>>(*this, m, ret);
			errProc(cudaGetLastError(), "fail to execute the kernel function in Matrix::operator*()");

			//�����
			errProc(cudaMemcpy(ret.hostData.get(), ret.deviceData, 
				ret.size()*sizeof(float), cudaMemcpyDeviceToHost),
					"could not read result in device in Matrix::operator*()");

			//����
			endCUDACompution();
			m.endCUDACompution();
			ret.endCUDACompution();
		}
		catch (std::exception e)
		{
			std::cerr << e.what();
			clearDeviceMem();
			m.clearDeviceMem();
			ret.clearDeviceMem();
		}
		return ret;
	}
};
//����˷���C=A*B
//ÿ��block����C�е�һ��
__global__ void matrixMul(Matrix a, Matrix b, Matrix c)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int bdim = blockDim.x;
	//tidƫ������ÿ�ε�������thread�ĸ���
	int i = 0;
	int w = c.widthDevice();
	int ha = a.heightDevice();
	//�������������������ѭ��
	if (w < bdim)
	{
		goto last;
	}
	//չ��ѭ��
#pragma unroll
	for (; i != w; i+=bdim)
	{
		//�����ʱ����
		float temp = 0;
		//
		int k = i + tid;
		for (int j = 0; j != ha; ++j)
		{
			temp += a.accDevice(bid, j)*b.accDevice(j, k);
		}
		c.accDevice(bid, k) = temp;
	}
last:
	int k = i + tid;
	//���ھ����С���ܲ�Ϊ256������������ʣ�ಿ�ֽ��м���
	if (k < w)
	{
		float temp = 0;
		for (int j = 0; j != ha; ++j)
		{
			temp += a.accDevice(bid, j)*b.accDevice(j, k);
		}
		c.accDevice(bid, k) = temp;
	}

}

Matrix multiplication(Matrix &a, Matrix &b)
{
	Matrix ret(a.height(), b.width());
	if (a.width() != b.height())
		return ret;
	for (int i = 0; i != ret.height(); ++i)
	{
		for (int j = 0; j != ret.width(); ++j)
		{
			float temp = 0;
			for (int k = 0; k != a.width(); ++k)
			{
				temp += a(i, k)*b(k, j);
			}
			ret(i, j) = temp;
		}
	}
	return ret;
}

int main(int argc, char **argv)
{
	Matrix a(10, 10);
	Matrix b(10, 10);
	Matrix c_g, c_c;

	a.randomize();
	b.randomize();
	c_g = a*b;
	c_c = multiplication(a, b);

	std::ofstream outa("a.txt");
	std::ofstream outb("b.txt");
	std::ofstream out("err.log");
	char buffer[200];
	for (int i = 0; i != c_c.width(); ++i)
	{
		for (int j = 0; j != c_c.height(); ++j)
		{
			outa << a(i, j) << ' ';
			outb << b(i, j) << ' ';
			if (abs(c_c(i, j) - c_g(i, j) > 1e-6))
			{
				sprintf(buffer, "(%d, %d), c_c:%f, c_g:%f\n", i, j, c_c(i, j), c_g(i, j));
				out << buffer << std::flush;
			}
		}
		outa << std::endl;
		outb << std::endl;
	}
	system("pause");
	return 0;
}
