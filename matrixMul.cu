
//
#include <cstdlib>
#include <exception>
#include <cstdio>
#include <utility>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>

#include "ran_timer.h"
#include "ran_helper_functions.h"

class Matrix;

__global__ void matrixMul(Matrix *a, Matrix *b, Matrix *c);

class Matrix
{
	//�������ڴ�������
	std::unique_ptr<float[]> hostData;
	//�������Դ������
	float *deviceData;
	//�������Դ��еĸ���
	Matrix *deviceMat;
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
		
		errProc(cudaMalloc(&deviceMat, sizeof(Matrix)));
		errProc(cudaMemcpy(deviceMat, this, sizeof(Matrix), cudaMemcpyHostToDevice));
	}
	void clearDeviceMem()
	{
		if (deviceData != nullptr)
		{
			cudaFree(deviceData);
			deviceData = nullptr;
		}
		if (deviceMat != nullptr)
		{
			cudaFree(deviceMat);
			deviceMat = nullptr;
		}
	}
	//����CUDA����
	void endCUDACompution()
	{
		clearDeviceMem();
	}

public:
	//kernel���в���
	static int threads;

	Matrix() :_width(1), _height(1), _size(1), 
		deviceData(nullptr), deviceMat(nullptr)
	{
		hostData = std::make_unique<float[]>(size());
		zeroing();
	}
	//wΪ��ȣ�hΪ�߶�
	Matrix(size_t w, size_t h) :_width(w), _height(h), _size(w*h), 
		deviceData(nullptr), deviceMat(nullptr)
	{
		hostData = std::make_unique<float[]>(size());
		zeroing();
	}
	//
	Matrix(Matrix& m) :_width(m.width()), _height(m.height()), _size(m.size()),
		deviceData(nullptr), deviceMat(nullptr)
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
	__host__ __device__ size_t width() const
	{
		return _width;
	}
	//����ĸ߶ȣ�������
	__host__ __device__ size_t height() const
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
/*
#ifdef _DEBUG
		if (id > size())
			throw std::exception("id out of range in Matrix::operator[]");
#endif
*/
		return hostData[id];
	}
	//ͨ����ά�±��������
	__host__ __device__ float &operator()(size_t x, size_t y) const
	{
/*
#ifdef _DEBUG

		if (x > _width)
			throw std::exception("x out of range in Matrix::operator()");
		if (y > _height)
			throw std::exception("y out of range in Matrix::operator()");
#endif
*/
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

			if (Matrix::threads > 256)
			{
				throw std::exception("too many threads in one block");
			}
			//ִ��
			matrixMul<<<(int)height(), threads>>>(deviceMat, m.deviceMat, ret.deviceMat);
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
__global__ void matrixMul(Matrix *a, Matrix *b, Matrix *c, int start, int end)
{
	//__shared__ float res[256];
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int bdim = blockDim.x;
	//tidƫ������ÿ��ѭ������thread�ĸ���
	int i = 0;
	int w = c->width();
	int ha = a->height();
	//�������������������ѭ��
	if (w < bdim)
	{
		goto last;
	}
	//ÿ��ѭ���������c�е�bid�е�bdim����
	for (; i < w; i += bdim)
	{
		//�����ʱ����
		float temp = 0;
		//
		int k = i + tid;
		for (int j = 0; j != ha; ++j)
		{
			temp += a->accDevice(bid, j)*b->accDevice(j, k);
		}
		c->accDevice(bid, k) = temp;
	}
last:
	int k = i + tid;
	//���ھ����С���ܲ�Ϊthread��������������ʣ�ಿ�ֽ��м���
	if (k < w)
	{
		float temp = 0;
		for (int j = 0; j != ha; ++j)
		{
			temp += a->accDevice(bid, j)*b->accDevice(j, k);
		}
		c->accDevice(bid, k) = temp;
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

int Matrix::threads = 32;

int main(int argc, char **argv)
{
	std::vector<int> n = { /*10,50,100, 200, 300, 400, 500,*/ 1000 };
	std::vector<int> thd = { 32,64,128,256 };
	bool enableCPUTest = false;
	bool enableGPUTest = true;
	bool enableCheck = false;

	Timer t;
	std::ofstream res("result.txt");

	char buffer[200];
	
	displayInfo(std::cout);

	res << "Data size";
	if (enableCPUTest)
		res << "CPU\t";

	res << buffer;
	for (auto i : thd)
	{
		res << i << '\t';
	}
	res << std::endl;

	for (int i = 0; i != n.size(); ++i)
	{
		std::cout << "data size = "<< n[i] << std::endl;
		res << n[i] << "\t";

		Matrix a(n[i], n[i]);
		Matrix b(n[i], n[i]);
		Matrix c_g, c_c;

		a.randomize();
		b.randomize();
		//CPU ���Բ���
		if (enableCPUTest)
		{
			std::cout << "\tCPU begin" << std::endl;
			t.begin();
			c_c = multiplication(a, b);
			t.end();
			res << t.time() << "\t";
			t.reset();
		}
		//GPU���Բ���
		if (enableGPUTest)
		{
			std::cout << "\tGPU begin" << std::endl;
			for (int j = 0; j != thd.size(); ++j)
			{
				std::cout << "\t\tthd = " << thd[j] << std::endl;

				Matrix::threads = thd[j];

				t.begin();
				c_g = a*b;
				t.end();
				res << t.time() << "\t";
				t.reset();
			}
			res << std::endl;
		}
		//����У��
		if (enableCheck)
		{
			sprintf(buffer, "%d_a.txt", n[i]);
			std::ofstream outa(buffer);
			sprintf(buffer, "%d_b.txt", n[i]);
			std::ofstream outb(buffer);
			sprintf(buffer, "%d_err_log.txt", n[i]);
			std::ofstream errlog(buffer);
			//�����ֵ
			float threshold = (float)1e-7;

			for (int i = 0; i != c_c.width(); ++i)
			{
				for (int j = 0; j != c_c.height(); ++j)
				{
					outa << a(i, j) << ' ';
					outb << b(i, j) << ' ';
					if (abs(c_c(i, j) - c_g(i, j) > threshold))
					{
						sprintf(buffer, "(%d, %d), c_c:%f, c_g:%f\n", i, j, c_c(i, j), c_g(i, j));
						errlog << buffer << std::endl;
					}
				}
				outa << std::endl;
				outb << std::endl;
			}
		}
	}

	std::cout << "Press Enter to exit." << std::flush;
	getchar();

	return 0;
}
