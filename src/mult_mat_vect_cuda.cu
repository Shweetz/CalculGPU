#include<stdio.h>
#include<math.h>  // for ceil()
#include<stdexcept>

#include"tools.h"
#include"common.h"


// CUDA debugging
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define gpuKernelExecErrChk() { gpuCheckKernelExecutionError( __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if( abort )
			exit(code);
	}
}

inline void gpuCheckKernelExecutionError( const char *file, int line)
{
	/**
	  Check for invalid launch argument, then force the host to wait
	  until the kernel stops and checks for an execution error.
	  The synchronisation can be eliminated if there is a subsequent blocking
	  API call like cudaMemcopy. In this case the cudaMemcpy call can return
	  either errors which occurred during the kernel execution or those from
	  the memory copy itself. This can be confusing for the beginner, so it is
	  recommended to use explicit synchronisation after a kernel launch during
	  debugging to make it easier to understand where problems might be arising.
	 */
	gpuAssert( cudaPeekAtLastError(), file, line);
	gpuAssert( cudaDeviceSynchronize(), file, line);	
}

//---------------------------------------------------------

// STUDENTS BEGIN

__global__ 
void kernelSpmvCSR(uint rowsNbr, const float *values, const uint *col_ind, const uint *row_ptr, const float *v, float *y)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Matrice:values[],col_ind[],row_ptr[]
	// Vecteur:v[]
	
	int row_beg = row_ptr[x];
	int row_end = row_ptr[x+1];
	
	// y[x] est dans la mémoire globale, on crée une variable dans la mémoire locale pour gagner du temps => utilisation d'un registre du GPU
	float dot = y[x];
	for (int i = row_beg; i < row_end; i++)
	{
		dot += values[i] * v[col_ind[i]];
	}
	y[x] = dot;
}

// STUDENTS END


//---------------------------------------------------------

/**
  Compute MxV on GPU. CSR method.
  A reference result can be passed to check that the computation is ok.
*/
Matrix* gpuSpmvCSR(const MatrixCSR *m, const Matrix *v, const Matrix *reference = NULL)
{
	const char *name = "CSR method on GPU";
	double gpuComputeTime = 0.0; // time measurement
	double gpuRunTime = 0.0; // time measurement

	if(m->w != v->h)
		throw std::runtime_error("Failed to multiply matrices, size mismatch.");
	if(v->w != 1)
		throw std::runtime_error("Failed to multiply matrices, vector size mismatch.");

	// output matrix size
	uint width = v->w;
	uint height = m->h;
	Matrix *mv = createMatrix(width, height);

	// data size
	uint valuesSizeInBytes = m->nzNbr * sizeof(float);
	uint col_indSizeInBytes = m->nzNbr * sizeof(uint);
	uint row_ptrSizeInBytes = (m->h + 1) * sizeof(uint);
	uint vSizeInBytes = (v->h) * sizeof(float);
	uint mvSizeInBytes = (m->h) * sizeof(float);

	// make a dummy memory allocation to wake up NVIDIA driver
	// before starting time measurement
	int *gpuWakeUp;
	gpuErrchk( cudaMalloc( (void**) &gpuWakeUp, 1) );

	// STUDENTS BEGIN
	
	// On va appeler la méthode kernelSpmvCSR mais d'abord il faut allouer les paramètres à passer
	
	float* valuesIn;
	gpuErrchk(cudaMalloc((void**) &valuesIn, valuesSizeInBytes));
	
	uint* colIndIn;
	gpuErrchk(cudaMalloc((void**) &colIndIn, col_indSizeInBytes));
	
	uint* rowPtrIn;
	gpuErrchk(cudaMalloc((void**) &rowPtrIn, row_ptrSizeInBytes));
	
	float* vIn;
	gpuErrchk(cudaMalloc((void**) &vIn, vSizeInBytes));
	
	float* mvOut;
	gpuErrchk(cudaMalloc((void**) &mvOut, mvSizeInBytes));
	
	// transfer data from CPU memory to GPU memory
	top(0); // start time measurement
	gpuErrchk(cudaMemcpy(valuesIn, m->data, valuesSizeInBytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(colIndIn, m->col_ind, col_indSizeInBytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(rowPtrIn, m->row_ptr, row_ptrSizeInBytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(vIn, v->data, vSizeInBytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(mvOut, mv->data, mvSizeInBytes, cudaMemcpyHostToDevice));
	
	dim3 dimBlock(32);
	dim3 dimGrid((int) ceilf(m->h*1.0/dimBlock.x));
	top(1);
	
	// Kernel launch
	kernelSpmvCSR<<<dimGrid, dimBlock>>>(m->h, valuesIn, colIndIn, rowPtrIn, vIn, mvOut);
	
	gpuKernelExecErrChk();
	gpuComputeTime = top(1); // pure computation duration
	
	gpuErrchk(cudaMemcpy(mv->data, mvOut, mvSizeInBytes, cudaMemcpyDeviceToHost));
	gpuRunTime = top(0); // computation and memory transfert time
	
	// release CPU memory
	cudaFree(valuesIn);
	cudaFree(colIndIn);
	cudaFree(rowPtrIn);
	cudaFree(vIn);
	cudaFree(mvOut);
	
	// STUDENTS END

	// check result, display run time if result is correct
	bool displayRunTime = true;
	if( reference )
	{
		if(! checkResult(name, reference, mv))
			displayRunTime = false;
	}

	if(displayRunTime)
		printf("%s: M(%dx%d)xV computed in %f ms (%f ms of pure computation).\n", name, m->w, m->h, gpuRunTime, gpuComputeTime);

	return mv;
}

//---------------------------------------------------------

// STUDENTS BEGIN

/*__global__ void kernelSpmvELL(uint rowsNbr, const float *values, const uint *col_ind, uint nzRowSz, const float *v, float *y)
{
	
}*/

// STUDENTS END

//---------------------------------------------------------

/**
  Compute MxV on GPU. ELL method.
  A reference result can be passed to check that the computation is ok.
*/
Matrix* gpuSpmvELL(const MatrixELL *m, const Matrix *v, const Matrix *reference = NULL)
{
	const char *name = "ELL method on GPU";
	double gpuComputeTime = 0.0; // time measurement
	double gpuRunTime = 0.0; // time measurement

	if(m->w != v->h)
		throw std::runtime_error("Failed to multiply matrices, size mismatch.");
	if(v->w != 1)
		throw std::runtime_error("Failed to multiply matrices, vector size mismatch.");

	// output matrix size
	uint width = v->w;
	uint height = m->h;
	Matrix *mv = createMatrix(width, height);

	// data size
	uint valuesSizeInBytes = m->nzRowSz * m->h * sizeof(float);
	uint col_indSizeInBytes = m->nzRowSz * m->h * sizeof(uint);
	uint vSizeInBytes = (v->h) * sizeof(float);
	uint mvSizeInBytes = (m->h) * sizeof(float);

	// make a dummy memory allocation to wake up NVIDIA driver
	// before starting time measurement
	int *gpuWakeUp;
	gpuErrchk( cudaMalloc( (void**) &gpuWakeUp, 1) );

	// STUDENTS BEGIN

	//TODO
	
	// STUDENTS END

	// check result, display run time if result is correct
	bool displayRunTime = true;
	if( reference )
	{
		if(! checkResult(name, reference, mv))
			displayRunTime = false;
	}

	if(displayRunTime)
		printf("%s: M(%dx%d)xV computed in %f ms (%f ms of pure computation).\n", name, m->w, m->h, gpuRunTime, gpuComputeTime);

	return mv;
}

//---------------------------------------------------------

// STUDENTS BEGIN

//kernelSpmvCSRVect(uint rowsNbr, const float *values, const uint *col_ind, const uint *row_ptr, const float *v, float *y)
//TODO

// STUDENTS END

//---------------------------------------------------------

/**
  Compute MxV on GPU. CSR-Vect method.
  A reference result can be passed to check that the computation is ok.
*/
Matrix* gpuSpmvCSRVect(const MatrixCSR *m, const Matrix *v, const Matrix *reference = NULL)
{
	const char *name = "CSR-Vect method on GPU";
	double gpuComputeTime = 0.0; // time measurement
	double gpuRunTime = 0.0; // time measurement

	if(m->w != v->h)
		throw std::runtime_error("Failed to multiply matrices, size mismatch.");
	if(v->w != 1)
		throw std::runtime_error("Failed to multiply matrices, vector size mismatch.");

	// output matrix size
	uint width = v->w;
	uint height = m->h;
	Matrix *mv = createMatrix(width, height);

	// data size
	uint valuesSizeInBytes = m->nzNbr * sizeof(float);
	uint col_indSizeInBytes = m->nzNbr * sizeof(uint);
	uint row_ptrSizeInBytes = (m->h + 1) * sizeof(uint);
	uint vSizeInBytes = (v->h) * sizeof(float);
	uint mvSizeInBytes = (m->h) * sizeof(float);

	// make a dummy memory allocation to wake up NVIDIA driver
	// before starting time measurement
	int *gpuWakeUp;
	gpuErrchk( cudaMalloc( (void**) &gpuWakeUp, 1) );

	// STUDENTS BEGIN

	//TODO
	
	// STUDENTS END

	// check result, display run time if result is correct
	bool displayRunTime = true;
	if( reference )
	{
		if(! checkResult(name, reference, mv))
			displayRunTime = false;
	}

	if(displayRunTime)
		printf("%s: M(%dx%d)xV computed in %f ms (%f ms of pure computation).\n", name, m->w, m->h, gpuRunTime, gpuComputeTime);

	return mv;
}
