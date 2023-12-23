#include <iostream>
#include <string>
#include <omp.h>
#include <stdio.h>
#include <windows.h>

using namespace std;

int main()
{
	cout << "Input threads count and matrix size" << endl;
	int threads, size;
	cin >> threads >> size;
	cout << omp_get_max_threads();


	omp_set_num_threads(threads);

	// инициализация матриц
	int* A = new int [size * size];
	int* B = new int[size * size];
	int* C = new int[size * size];

	// заполнение матриц
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			A[i * size + j] = B[i * size + j] = i * j;

	double start = omp_get_wtime();

	int i, j, k;
	// основной вычислительный блок
#pragma omp parallel for shared(A, B, C) private(i, j, k)
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			C[i * size + j] = 0;
			for (k = 0; k < size; k++)
				C[i * size + j] += A[i* size + k] * B[k * size + j];
		}
	}

	double end = omp_get_wtime();

	cout << "Size: " << size << endl;
	cout << "Duration: " << end - start;

	return 0;
}