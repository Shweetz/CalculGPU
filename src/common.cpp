#include "common.h"

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <limits.h>


//---------------------------------------------------------

Matrix* createMatrix( uint w, uint h)
{
	Matrix *m = (Matrix*) malloc(sizeof(Matrix));
	m->w = w;
	m->h = h;
	m->data = (float*) calloc(w*h, sizeof(float));

	if( m->data == NULL )
	{
		deleteMatrix(&m);
		throw std::runtime_error("Failed to allocate memory for matrix.");
	}

	return m;
}

//---------------------------------------------------------

void deleteMatrix(Matrix **m)
{
	free((*m)->data);
	free(*m);
	*m = NULL;
}

//---------------------------------------------------------

void writeMatrixToFile( const float *data, uint width, uint height, const char *fileName)
{
	std::ofstream file(fileName);
	if( ! file.is_open() )
		throw std::runtime_error( std::string("Failed to open file '") + fileName + "' !\n");

	file.precision(15);

	file << width << " " << height << "\n";
	for( uint r = 0; r < height; r++)
	{
		for(uint c = 0; c < width; c++)
			file << data[r*width + c] << " ";

		file << "\n";
	}
}

//---------------------------------------------------------

void writeMatrixToFile( const Matrix *m, const char *fileName)
{
	return writeMatrixToFile( m->data, m->w, m->h, fileName);
}

//---------------------------------------------------------

Matrix* readMatrixFromFile(const char *fileName)
{
	std::ifstream file(fileName);
	if( ! file.is_open() )
		throw std::runtime_error(std::string("Failed to open file '") + fileName + "' !\n");

	uint width = 0;
	uint height = 0;
	file >> width >> height;
	Matrix *m = createMatrix( width, height);

	for( uint r = 0; r < height; r++)
		for(uint c = 0; c < width; c++)
			file >> m->data[r*width + c];

	return m;
}

//---------------------------------------------------------

void initMatrix( uint width, uint height, float *data, float sparseRate)
{
	// initialize with random data in [1,100]
	for( uint r = 0; r < height; r++)
		for( uint c = 0; c < width; c++)
			data[r*width + c] = 1 + rand() % 99;
	
	// set some values to zero according to sparseRate
	if( sparseRate > 0.0f )
	{
		uint zeroNbr = width * height * sparseRate;

		for( uint i = 0; i < zeroNbr; i++)
		{
			uint r = rand() % height;
			uint c = rand() % width;

			if( data[r*width + c] == 0.0f )
			{
				// the value is already 0, repeat
				i--;
				continue;
			}

			data[r*width + c] = 0.0f;
		}
	}
}

//---------------------------------------------------------

void initMatrix( const Matrix *m, float sparseRate)
{
	initMatrix( m->w, m->h, m->data, sparseRate);
}

//---------------------------------------------------------

bool areEqual( uint width, uint height, const float *m1, const float *m2)
{
	uint diffCnt = 0;
	for( uint r = 0; r < height; r++)
	{
		for( uint c = 0; c < width; c++)
		{
			if( m1[r*width + c] != m2[r*width + c] )
			{
				diffCnt++;
				// display the first differences
				if( diffCnt <= 5 )
					printf("! at r=%d, c=%d  m1=%f  m2=%f\n", r, c, m1[r*width + c], m2[r*width + c]);
			}
		}
	}
	
	if( diffCnt == 0 )
		return true;
	else
		return false;
}

//---------------------------------------------------------

bool areEqual( const Matrix *m1, const Matrix *m2)
{
	return areEqual( m1->w, m1->h, m1->data, m2->data);
}

//---------------------------------------------------------

void printMatrix( const Matrix *m, const char *title)
{
	uint maxWidth = 6;
	uint maxHeight = 10;
	uint width = m->w;
	uint height = m->h;
	float *data = m->data;

	printf( "%s:\n", title);

	for( uint r = 0; r < height; r++)
	{
		if( r >= maxHeight )
		{
			printf("  ...\n");
			break;
		}

		for( uint c = 0; c < width; c++)
		{
			if( c >= maxWidth )
			{
				printf("  ...");
				break;
			}

			printf( "  %.5g", data[r*width + c]);
		}

		printf("\n");
	}
}

//---------------------------------------------------------

MatrixCSR* matrixToCSR(const Matrix *m)
{
	// temporary, dynamically growing, storage
	std::vector<float> data;
	std::vector<uint> row_ptr;
	std::vector<uint> col_ind;

	// collect non zero values
	for(uint r = 0; r < m->h; r++)
	{
		row_ptr.push_back(UINT_MAX);

		for(uint c = 0; c < m->w; c++)
		{
			float value = m->data[r*(m->w) + c];
			if(value != 0.0f)
			{
				data.push_back(value);
				col_ind.push_back(c);

				if(row_ptr[r] == UINT_MAX)
					row_ptr[r] = data.size()-1; // index of the last non zero value
			}
		}
	}

	// the last row_ptr element is the number of non zero values
	row_ptr.push_back(data.size());

	// parse row_ptr array backward to fix empty lines
	for(int i = row_ptr.size()-1; i >= 0; i--)
	{
		if(row_ptr[i] == UINT_MAX)
			row_ptr[i] = row_ptr[i+1];
			// an empty line starts at the same index than
			// the next non-empty line
	}

	// allocate memory to store CSR matrix
	MatrixCSR *mCSR = (MatrixCSR*) malloc(sizeof(MatrixCSR));
	size_t dataBytes = data.size() * sizeof(float);
	size_t col_indBytes = col_ind.size() * sizeof(uint);
	size_t row_ptrBytes = row_ptr.size() * sizeof(uint);
	mCSR->data = (float*) malloc(dataBytes);
	mCSR->col_ind = (uint*) malloc(col_indBytes);
	mCSR->row_ptr = (uint*) malloc(row_ptrBytes);

	// populate CSR matrix
	mCSR->w = m->w;
	mCSR->h = m->h;
	mCSR->nzNbr = data.size();
	memcpy(mCSR->data, &(data[0]), dataBytes);
	memcpy(mCSR->col_ind, &(col_ind[0]), col_indBytes);
	memcpy(mCSR->row_ptr, &(row_ptr[0]), row_ptrBytes);

	return mCSR;
}

//---------------------------------------------------------

void deleteMatrixCSR(MatrixCSR **mCSR)
{
	free((*mCSR)->data);
	free((*mCSR)->col_ind);
	free((*mCSR)->row_ptr);
	free(*mCSR);
	*mCSR = NULL;
}

//---------------------------------------------------------

void printMatrixCSR( const MatrixCSR *mCSR, const char *title)
{
	printf( "%s:\n", title);

	// print non zero values
	printf("  values = [");
	for( uint i = 0; i < mCSR->nzNbr; i++)
		printf( " %.5g", mCSR->data[i]);
	printf(" ]\n");

	// print column indexes
	printf("  col_ind = [");
	for( uint i = 0; i < mCSR->nzNbr; i++)
		printf( " %d", mCSR->col_ind[i]);
	printf(" ]\n");

	// print rows pointers
	printf("  row_ptr = [");
	for( uint i = 0; i <= mCSR->h; i++)
		printf( " %d", mCSR->row_ptr[i]);
	printf(" ]\n");
}

//---------------------------------------------------------

bool checkResult(const char *title, const Matrix *reference, const Matrix *result)
{
	// check that result is identical to reference
	bool retVal = true;
	if( areEqual(result, reference) )
		printf("%s: results are good.\n", title);
	else
	{
		printf("%s: results are BAD !!!\n", title);
		retVal = false;
	}

	return retVal;
}

//---------------------------------------------------------

MatrixELL* matrixToELL(const Matrix *m)
{
	// parse matrix to find the line with the maximum
	// number of non zero values
	uint maxNzNbr = 0;
	for(uint r = 0; r < m->h; r++)
	{
		uint nzNbr = 0;
		for(uint c = 0; c < m->w; c++)
		{
			float value = m->data[r*(m->w) + c];
			if(value != 0.0f)
				nzNbr++;
		}

		if(nzNbr > maxNzNbr)
			maxNzNbr = nzNbr;
	}

	// allocate memory to store ELL matrix
	MatrixELL *mELL = (MatrixELL*) malloc(sizeof(MatrixELL));
	mELL->data = (float*) calloc(m->h * maxNzNbr, sizeof(float));
	mELL->col_ind = (uint*) calloc(m->h * maxNzNbr, sizeof(uint));

	// populate ELL matrix
	mELL->w = m->w;
	mELL->h = m->h;
	mELL->nzRowSz = maxNzNbr;
	for(uint r = 0; r < m->h; r++)
	{
		// non-zero values are stored in COLUMN MAJOR order
		float *rowPtr = &(mELL->data[r]);
		uint *col_indPtr = &(mELL->col_ind[r]);

		for(uint c = 0; c < m->w; c++)
		{
			float value = m->data[r*(m->w) + c];
			if(value != 0.0f)
			{
				*rowPtr = value;
				rowPtr += mELL->h;
				*col_indPtr = c;
				col_indPtr += mELL->h;
			}
		}
	}

	return mELL;
}

//---------------------------------------------------------

void deleteMatrixELL(MatrixELL **mELL)
{
	free((*mELL)->data);
	free((*mELL)->col_ind);
	free(*mELL);
	*mELL = NULL;
}

//---------------------------------------------------------

void printMatrixELL( const MatrixELL *mELL, const char *title)
{
	printf( "%s:\n", title);

	// print non zero values
	printf("  values = [");
	uint nzNbr = mELL->h * mELL->nzRowSz;
	for( uint i = 0; i < nzNbr; i++)
		printf( " %.5g", mELL->data[i]);
	printf(" ]\n");

	// print column indexes
	printf("  col_ind = [");
	for( uint i = 0; i < nzNbr; i++)
		printf( " %d", mELL->col_ind[i]);
	printf(" ]\n");
}

