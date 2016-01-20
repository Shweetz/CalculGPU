
Matrix* gpuSpmvCSR(const MatrixCSR *m, const Matrix *v, const Matrix *reference = NULL);
Matrix* gpuSpmvELL(const MatrixELL *m, const Matrix *v, const Matrix *reference = NULL);
Matrix* gpuSpmvCSRVect(const MatrixCSR *m, const Matrix *v, const Matrix *reference = NULL);
