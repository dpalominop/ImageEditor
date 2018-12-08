#ifndef _MY_USER_TYPES
#define _MY_USER_TYPES

#include <string>
#include <cstring>
#include <iostream>
//#define ERROR_CHECK

static float abs1f(const float src)
{
    return (src < 0.0f) ? -src : src;
}

typedef struct
{
    float r;
    float g;
    float b;
} c3f;

typedef struct
{
    int x;
    int y;
} p2i;

typedef struct
{
    float x;
    float y;
} p2f;

template <class T>
class rawArray2D
{
public:
	T *data;
	unsigned int width;
	unsigned int height;
	rawArray2D(const unsigned int h, const unsigned int w);
	rawArray2D();
	rawArray2D(const rawArray2D<T>& src);
	rawArray2D(rawArray2D<T>&& src);
	~rawArray2D();
	void resize_without_copy(const unsigned int h, const unsigned int w);
    T& operator()(const unsigned int y, const unsigned int x) const;
    T& operator[](const unsigned int offset) const;
	rawArray2D<T> & operator=(const rawArray2D<T>& src);
	rawArray2D<T> & operator=(rawArray2D<T>&& src);

	int load_from_file(const std::string & path);

	int store_to_file(const std::string & path);
};

template <class T>
static rawArray2D<T> operator*(const rawArray2D<T>& A, const rawArray2D<T>& B)
{
	rawArray2D<T> C(B.height, A.width);
	if (A.width != B.height)
	{
		std::string err("matrix multiplication error: size mismatch");
#ifdef ERROR_CHECK
		throw(err);
#endif
		std::cout << err << std::endl;
	}
	for (unsigned int i = 0; i < A.height; ++i)
	for (unsigned int j = 0; j < B.width; ++j)
	{
		T val = 0;
		for (unsigned int r = 0; r < A.width; ++r)
			val += A(i, r)*B(r, j);
		C(i, j) = val;
	}
	return C;
}

template <class T>
rawArray2D<T>::rawArray2D(const unsigned int h, const unsigned int w)
{
	height = h;
	width = w;
    std::cout << sizeof(T)*h*w << std::endl;
	data = new T[w*h];
}

template <class T>
rawArray2D<T>::rawArray2D()
{
	height = 0;
	width = 0;
	data = nullptr;
}

template <class T>
rawArray2D<T>::rawArray2D(const rawArray2D<T>& src)
{
	const unsigned int sz = src.width*src.height;
	width = src.width;
	height = src.height;
	data = new T[sz];
	memcpy(data, src.data, sz*sizeof(T));
}

template <class T>
rawArray2D<T>::rawArray2D(rawArray2D<T>&& src)
{
	width = src.width;
	height = src.height;
	data = src.data;
	////
	src.width = 0;
	src.height = 0;
	src.data = nullptr;
	//std::cout << "move constructor " << width << "x" << height << std::endl;
}

template <class T>
rawArray2D<T>::~rawArray2D()
{
	if (data != nullptr)
		delete[] data;
}

template <class T>
void rawArray2D<T>::resize_without_copy(const unsigned int h, const unsigned int w)
{
	if (data != nullptr) delete[] data;
	width = w;
	height = h;
	data = new T[w*h];
}

template <class T>
T& rawArray2D<T>::operator()(const unsigned int y, const unsigned int x) const
{
#ifdef ERROR_CHECK
    if (x < 0) { std::cout<<"x="<<x<<"<0"<<std::endl; throw (std::string("x below 0")); }
    if (y < 0) { std::cout<<"y="<<y<<"<0"<<std::endl; throw (std::string("y below 0")); }
    if (x >= width) { std::cout << "x=" << x << "width=" << width<<std::endl; throw(std::string("x greater than array width")); }
    if (y >= height) { std::cout << "y=" << y << "height=" << height<<std::endl; throw(std::string("y greater than array height")); }
#endif
	return (data[y*width + x]);
}

template <class T>
T& rawArray2D<T>::operator[](const unsigned int offset) const
{
	return (data[offset]);
}

template <class T>
rawArray2D<T> & rawArray2D<T>::operator=(const rawArray2D<T>& src)
{
	if (&src == this) return *this;
	const unsigned int sz = src.width*src.height;
	if ((src.width != width) || (src.height != height))
	{
		if (data != nullptr)
			delete[] data;
		width = src.width;
		height = src.height;
		data = new T[sz];
	}
	memcpy(data, src.data, sz*sizeof(T));
	return *this;
}

template <class T>
rawArray2D<T> & rawArray2D<T>::operator=(rawArray2D<T>&& src)
{
	if (&src == this) return *this;
	width = src.width;
	height = src.height;
	if (data != nullptr)
		delete[] data;
	data = src.data;
	///
	src.width = 0;
	src.height = 0;
	src.data = nullptr;
	//std::cout << "move= " << width<<"x"<<height<< std::endl;
	return *this;
}

template <class T>
int rawArray2D<T>::load_from_file(const std::string & path)
{
	// determne file size
	FILE *const f = fopen(path.c_str(), "rb");
	fseek(f, 0, SEEK_END);

	// now read file to array
	fseek(f, 0, SEEK_SET);
	if (fread(&height, sizeof(unsigned int), 1, f) < 1)
		return -1;
	if (fread(&width, sizeof(unsigned int), 1, f) < 1)
		return -1;
	const int datasize = height*width;
	resize_without_copy(height, width);
	int res = fread(&data[0], sizeof(T), datasize, f);
	if (res < datasize)
	{
		std::cout << "no data in file" << std::endl;
		fclose(f);
		return -1;
	}
	fclose(f);
	return 0;
}

template <class T>
int rawArray2D<T>::store_to_file(const std::string & path)
{
	FILE *const f = fopen(path.c_str(), "wb");

	if (fwrite(&height, sizeof(unsigned int), 1, f) < 1)
		return -1;
	if (fwrite(&width, sizeof(unsigned int), 1, f) < 1)
		return -1;
	const int res = fwrite(&data[0], sizeof(T), width*height, f);
	if (res < 4)
	{
		std::cout << "no data in file" << std::endl;
		fclose(f);
		return -1;
	}
	fclose(f);
	return 0;
}

#ifdef ERROR_CHECK
#undef ERROR_CHECK
#endif

#endif //_MY_USER_TYPES
