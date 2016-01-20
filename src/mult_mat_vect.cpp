#include<stdio.h>
#include<stdexcept>

#include"tools.h"
#include"common.h"
#include"mult_mat_vect_cuda.h"


/**
  Compute M1xM2 on CPU. Classical method.
*/
Matrix* cpuSpmvClassical(const Matrix *m1, const Matrix *m2)
{
	if(m1->w != m2->h)
		throw std::runtime_error("Failed to multiply matrices, size mismatch.");

	// output matrix size
	uint width = m2->w;
	uint height = m1->h;
	Matrix *m1xm2 = createMatrix(width, height);

	top(0);
	for(uint r = 0; r < height; r++)
	{
		for(uint c = 0; c < width; c++)
		{
			float tmp = 0;

			for(uint k = 0; k < m1->w; k++)
				tmp += m1->data[r*(m1->w) + k] * m2->data[k*(m2->w) + c];

			m1xm2->data[r*width + c] = tmp;
		}
	}
	double cpuRunTime = top(0);

	printf("Classical method on cpu: M(%dx%d)xV computed in %f ms.\n", m1->w, m1->h, cpuRunTime);

	return m1xm2;
}


/**
  Compute MxV on CPU. CSR method.
  A reference result can be passed to check that the computation is ok.
*/
Matrix* cpuSpmvCSR(const MatrixCSR *m, const Matrix *v, const Matrix *reference = NULL)
{
	const char *name = "CSR method on cpu";

	if(m->w != v->h)
		throw std::runtime_error("Failed to multiply matrices, size mismatch.");
	if(v->w != 1)
		throw std::runtime_error("Failed to multiply matrices, vector has more than 1 column.");

	// output matrix size
	uint width = v->w;
	uint height = m->h;
	Matrix *mv = createMatrix(width, height);

	top(0);
	for(uint r = 0; r < height; r++)
	{
		float dot = 0.0f;

		uint row_beg = m->row_ptr[r];
		uint row_end = m->row_ptr[r+1];

		for(uint i = row_beg; i < row_end; i++)
			dot += m->data[i] * v->data[m->col_ind[i]];

		mv->data[r] = dot;
	}
	double cpuRunTime = top(0);

	// check result, display run time if result is correct
	bool displayRunTime = true;
	if( reference )
	{
		if(! checkResult(name, reference, mv))
			displayRunTime = false;
	}

	if(displayRunTime)
		printf("%s: M(%dx%d)xV computed in %f ms.\n", name, m->w, m->h, cpuRunTime);

	return mv;
}


/**
  Compute MxV on CPU. ELL method.
  A reference result can be passed to check that the computation is ok.
*/
Matrix* cpuSpmvELL(const MatrixELL *m, const Matrix *v, const Matrix *reference = NULL)
{
	const char *name = "ELL method on cpu";

	if(m->w != v->h)
		throw std::runtime_error("Failed to multiply matrices, size mismatch.");
	if(v->w != 1)
		throw std::runtime_error("Failed to multiply matrices, vector has more than 1 column.");

	// output matrix size
	uint width = v->w;
	uint height = m->h;
	Matrix *mv = createMatrix(width, height);

	top(0);

	// REMOVE FOR STUDENTS BEGIN

	//TODO
		
	// REMOVE FOR STUDENTS END

	double cpuRunTime = top(0);

	// check result, display run time if result is correct
	bool displayRunTime = true;
	if( reference )
	{
		if(! checkResult(name, reference, mv))
			displayRunTime = false;
	}

	if(displayRunTime)
		printf("%s: M(%dx%d)xV computed in %f ms.\n", name, m->w, m->h, cpuRunTime);

	return mv;
}


/**
  Do matrix-vector multiplication with various methods.
*/
int main(int argc, const char **argv)
{
	if(argc != 2)
	{
		printf("Usage: %s dataset_basename\n", argv[0]);
		printf("Example: %s  mat_1000x1500_0.50\n", argv[0]);
		return 1;
	}

	std::string matrixFileName = std::string(argv[1]) + ".M";
	std::string vectorFileName = std::string(argv[1]) + ".V";

	Matrix *m = readMatrixFromFile(matrixFileName.c_str());
	Matrix *v = readMatrixFromFile(vectorFileName.c_str());

	// classical method on CPU, used as reference
	Matrix *mv_cpu_classical = cpuSpmvClassical(m, v);

	// CSR method on CPU
	MatrixCSR *mCSR = matrixToCSR(m);
	//printMatrixCSR(mCSR, "mCSR");
	Matrix *mv_cpu_csr = cpuSpmvCSR(mCSR, v, mv_cpu_classical);
	deleteMatrix(&mv_cpu_csr);

	// CSR method on GPU
	Matrix *mv_gpu_csr = gpuSpmvCSR(mCSR, v, mv_cpu_classical);
	deleteMatrix(&mv_gpu_csr);

	// ELL method on CPU
	MatrixELL *mELL = matrixToELL(m);
	//printMatrixELL(mELL, "mELL");
	Matrix *mv_cpu_ell = cpuSpmvELL(mELL, v, mv_cpu_classical);
	deleteMatrix(&mv_cpu_ell);

	// ELL method on GPU
	Matrix *mv_GPU_ell = gpuSpmvELL(mELL, v, mv_cpu_classical);
	deleteMatrix(&mv_GPU_ell);

	// CSR-Vect method on GPU
	Matrix *mv_gpu_csr_vect = gpuSpmvCSRVect(mCSR, v, mv_cpu_classical);
	deleteMatrix(&mv_gpu_csr_vect);

	// release memory
	deleteMatrix(&m);
	deleteMatrix(&v);
	deleteMatrix(&mv_cpu_classical);
	deleteMatrixCSR(&mCSR);
	deleteMatrixELL(&mELL);

	return 0;
}

