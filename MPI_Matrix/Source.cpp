#include <stdio.h>
#include <mpi.h>
#include <iostream>
#include <string>

using namespace std;

int ProcRank, ProcNum;

void MatrixMultiplicationMPI(double*& A, double*& B, double*& C, int& Size) {
	int dim = Size;
	int i, j, k, p, ind;
	double temp;
	MPI_Status Status;
	int ProcPartSize = dim / ProcNum;
	int ProcPartElem = ProcPartSize * dim;
	double* bufA = new double[ProcPartElem];
	double* bufB = new double[ProcPartElem];
	double* bufC = new double[ProcPartElem];
	int ProcPart = dim / ProcNum, part = ProcPart * dim;

	MPI_Scatter(A, part, MPI_DOUBLE, bufA, part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(B, part, MPI_DOUBLE, bufB, part, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	temp = 0.0;
	for (i = 0; i < ProcPartSize; i++) {
		for (j = 0; j < ProcPartSize; j++) {
			for (k = 0; k < dim; k++)
				temp += bufA[i * dim + k] * bufB[j * dim + k];
			bufC[i * dim + j + ProcPartSize * ProcRank] = temp;
			temp = 0.0;
		}
	}

	int NextProc; int PrevProc;
	for (p = 1; p < ProcNum; p++) {
		NextProc = ProcRank + 1;
		if (ProcRank == ProcNum - 1)
			NextProc = 0;
		PrevProc = ProcRank - 1;
		if (ProcRank == 0)
			PrevProc = ProcNum - 1;
		MPI_Sendrecv_replace(bufB, part, MPI_DOUBLE, NextProc, 0, PrevProc, 0, MPI_COMM_WORLD, &Status);
		temp = 0.0;
		for (i = 0; i < ProcPartSize; i++) {
			for (j = 0; j < ProcPartSize; j++) {
				for (k = 0; k < dim; k++) {
					temp += bufA[i * dim + k] * bufB[j * dim + k];
				}
				if (ProcRank - p >= 0)
					ind = ProcRank - p;
				else ind = (ProcNum - p + ProcRank);
				bufC[i * dim + j + ind * ProcPartSize] = temp;
				temp = 0.0;
			}
		}
	}

	MPI_Gather(bufC, ProcPartElem, MPI_DOUBLE, C, ProcPartElem, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	delete[]bufA;
	delete[]bufB;
	delete[]bufC;
}


int main(int argc, char** argv) {

	int size = stoi(argv[1]);


	double* A = new double[size * size];
	double* B = new double[size * size];
	double* C = new double[size * size];

	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			A[i * size + j] = i * j;
			B[i * size + j] = i * j;
		}
	}

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	double start = MPI_Wtime();
	MatrixMultiplicationMPI(A, B, C, size);
	double end = MPI_Wtime();
	MPI_Finalize();
	

	if (!ProcRank) {
		cout << "Size: " << argv[1] << endl;
		cout << "Duration: " << end - start;
	}
	return 0;
}