
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
	//矩阵在内存中数据
	std::unique_ptr<float[]> hostData;
	//矩阵在显存的数据
	float *deviceData;
	//矩阵在显存中的复制
	Matrix *deviceMat;
	//宽度，每行的数据数
	size_t _width;
	//高度，行数
	size_t _height;
	//数据量
	size_t _size;

	//初始化CUDA，复制数据
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
	//结束CUDA计算
	void endCUDACompution()
	{
		clearDeviceMem();
	}

public:
	//kernel运行参数
	static int threads;

	Matrix() :_width(1), _height(1), _size(1), 
		deviceData(nullptr), deviceMat(nullptr)
	{
		hostData = std::make_unique<float[]>(size());
		zeroing();
	}
	//w为宽度，h为高度
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
	//矩阵的宽度，即列数
	__host__ __device__ size_t width() const
	{
		return _width;
	}
	//矩阵的高度，即行数
	__host__ __device__ size_t height() const
	{
		return _height;
	}
	//矩阵的数据数
	size_t size() const
	{
		return _size;
	}
	//随机化矩阵数据
	void randomize()
	{
		for (size_t i = 0; i != size(); ++i)
		{
			hostData[i] = rand<float>();
		}
	}
	//矩阵数据归零
	void zeroing()
	{
		for (size_t i = 0; i != size(); ++i)
		{
			hostData[i] = 0;
		}
	}
	//通过一维下标访问数据
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
	//通过二维下标访问数据
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
	//等于运算符重载
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
	//右值
	Matrix &operator=(Matrix&& m)
	{
		_height = m.height();
		_width = m.width();
		_size = m.size();

		hostData = std::move(m.hostData);
		return *this;
	}
	//访问显存中的数据
	__device__ float &accDevice(size_t x, size_t y)
	{
		return deviceData[x + y * _width];
	}
	//矩阵乘法
	Matrix operator*(Matrix &m)
	{
		Matrix ret(this->height(), m.width());
		try
		{
			//形状不对
			if (width() != m.height())
				throw std::exception("the shape of matrix does not satisfy\
					 the requirement of matrix multiplication in Matrix::operator*()");
			//初始化，复制到显存
			initCUDACompution();
			m.initCUDACompution();
			ret.initCUDACompution();

			if (Matrix::threads > 256)
			{
				throw std::exception("too many threads in one block");
			}
			//执行
			matrixMul<<<(int)height(), threads>>>(deviceMat, m.deviceMat, ret.deviceMat);
			errProc(cudaGetLastError(), "fail to execute the kernel function in Matrix::operator*()");

			//读结果
			errProc(cudaMemcpy(ret.hostData.get(), ret.deviceData, 
				ret.size()*sizeof(float), cudaMemcpyDeviceToHost),
					"could not read result in device in Matrix::operator*()");

			//清理
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
//矩阵乘法：C=A*B
//每个block计算C中的一行
__global__ void matrixMul(Matrix *a, Matrix *b, Matrix *c, int start, int end)
{
	//__shared__ float res[256];
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int bdim = blockDim.x;
	//tid偏移量，每次循环增加thread的个数
	int i = 0;
	int w = c->width();
	int ha = a->height();
	//如果数据量较少则跳过循环
	if (w < bdim)
	{
		goto last;
	}
	//每次循环计算矩阵c中第bid行的bdim个数
	for (; i < w; i += bdim)
	{
		//结果临时变量
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
	//由于矩阵大小可能不为thread数的整数倍，对剩余部分进行计算
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
		//CPU 测试部分
		if (enableCPUTest)
		{
			std::cout << "\tCPU begin" << std::endl;
			t.begin();
			c_c = multiplication(a, b);
			t.end();
			res << t.time() << "\t";
			t.reset();
		}
		//GPU测试部分
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
		//数据校验
		if (enableCheck)
		{
			sprintf(buffer, "%d_a.txt", n[i]);
			std::ofstream outa(buffer);
			sprintf(buffer, "%d_b.txt", n[i]);
			std::ofstream outb(buffer);
			sprintf(buffer, "%d_err_log.txt", n[i]);
			std::ofstream errlog(buffer);
			//误差阈值
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
