#include "./utils.h"

float cache;
int return_cache;
int nn_threads;

/**!
 * Creates a matrix filled with zeros. 
 * @param dims An array of matrix dimensions (int)[rows,columns] 
 * @result A pointer to the created matrix. 
 * @return A pointer to the created matrix. 
*/
dARRAY * zeros(int * dims){
  dARRAY * matrix = (dARRAY*)malloc(sizeof(dARRAY));
  matrix->matrix = (float*)calloc(dims[0]*dims[1],sizeof(float));
  matrix->shape[0] = dims[0];
  matrix->shape[1] = dims[1];
  return matrix;
}

/**!
 * Creates a matrix filled with ones. 
 * @param dims An array of matrix dimensions (int)[rows,columns] 
 * @result A pointer to the created matrix. 
 * @return A pointer to the created matrix. 
*/
dARRAY * ones(int * dims){
  dARRAY * matrix = (dARRAY*)malloc(sizeof(dARRAY));
  matrix->matrix = (float*)malloc(sizeof(float)*(dims[0]*dims[1]));
  omp_set_num_threads(nn_threads);
  #pragma omp parallel for num_threads(nn_threads)
  for(int i=0;i<dims[0]*dims[1];i++){
     matrix->matrix[i]=1;
  }
  matrix->shape[0] = dims[0];
  matrix->shape[1] = dims[1];
  return matrix;
}

/**!
 * Creates an identity matrix. 
 * @param dims An array of matrix dimensions (int)[rows,columns] 
 * @result A pointer of identity matrix. 
 * @return A pointer of identity matrix. 
*/
dARRAY * eye(int * dims){
  dARRAY * matrix = (dARRAY*)malloc(sizeof(dARRAY));
  matrix->matrix = (float*)calloc((dims[0]*dims[1]),sizeof(float));
  omp_set_num_threads(nn_threads);
  #pragma omp parallel for num_threads(nn_threads) collapse(1)
  for(int i=0;i<dims[0]; i++){
    for(int j=0;j<dims[1];j++)
      matrix->matrix[i*dims[1]+j] = i==j ? 1: 0;
  }
  matrix->shape[0] = dims[0];
  matrix->shape[1] = dims[1];
  return matrix;
}

/**!
 * Finds the transpose of the given matrix. 
 * @param Matrix The input Matrix of dARRAY Object 
 * @result A pointer to the result of Transpose(Matrix) 
 * @return A pointer to the result of Transpose(Matrix) 
*/
dARRAY * transpose(dARRAY * restrict Matrix){
  if(Matrix==NULL){
    printf("\033[1;31mError:\033[93m Matrix is empty. Call transpose() only after intializing dARRAY object.\033[0m\n");
    return NULL;
  }
  if(Matrix->shape[0]==1 && Matrix->shape[1]==1) return Matrix;
  dARRAY * matrix = (dARRAY*)malloc(sizeof(dARRAY));
  matrix->matrix = (float*)calloc(Matrix->shape[0]*Matrix->shape[1],sizeof(float));
  #pragma omp task
  cblas_somatcopy(CblasRowMajor,CblasTrans,Matrix->shape[0],Matrix->shape[1],1,Matrix->matrix,Matrix->shape[1],matrix->matrix,Matrix->shape[0]);
  matrix->shape[0] = Matrix->shape[1];
  matrix->shape[1] = Matrix->shape[0];
  return matrix;
}



dARRAY * transposem(dARRAY * restrict Matrix){
  if(Matrix==NULL){
    printf("\033[1;31mError:\033[93m Matrix is empty. Call transpose() only after intializing dARRAY object.\033[0m\n");
    return NULL;
  }
  if(Matrix->shape[0]==1 && Matrix->shape[1]==1) return Matrix;
  dARRAY * matrix = (dARRAY*)malloc(sizeof(dARRAY));
  matrix->matrix = (float*)calloc(Matrix->shape[0]*Matrix->shape[1],sizeof(float));
  omp_set_num_threads(2);
  double start = omp_get_wtime();
  #pragma omp parallel for num_threads(nn_threads) shared(Matrix,matrix) schedule(static)
  for(int i=0;i<Matrix->shape[0];i++)
    for(int j=0;j<Matrix->shape[1];j++)
      matrix->matrix[j*Matrix->shape[0]+i] = Matrix->matrix[i*Matrix->shape[1]+j];
  double end = omp_get_wtime();
  printf("Transpose took %lf seconds\n",(end-start));
  matrix->shape[0] = Matrix->shape[1];
  matrix->shape[1] = Matrix->shape[0];
  return matrix;
}

/**!
 * Finds the dot product (Matrix Multiplication) of two matrices. 
 * @param MatrixA First Matrix (float * __restrict__) 
 * @param MatrixB Second Matrix (float * __restrict__) 
 * @result Returns a pointer to the result of dot(MatrixA,MatrixB) 
 * @return A pointer to the result of dot(MatrixA,MatrixB) 
*/
dARRAY * dot(dARRAY * MatrixA, dARRAY * MatrixB){
  if(MatrixA->shape[1]!=MatrixB->shape[0]){
    printf("\033[1;31mError:\033[93m Shape error while performing dot(). Matrix dimensions do not align. %d(dim1) != %d(dim0)\033[0m\n",MatrixA->shape[1],MatrixB->shape[0]);
    return NULL;
  }
  if(MatrixB == NULL || MatrixA == NULL){
    printf("\033[1;31mError:\033[93m One of the input matrices is empty. Call dot() only after initializing dARRAY object\033[0m\n");
    return NULL;
  }
  long long int m,n,k;
  m = MatrixA->shape[0];
  n = MatrixB->shape[1];
  k = MatrixB->shape[0];
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (float*)calloc(m*n,sizeof(float));
  #pragma omp task
  cblas_sgemm(CblasRowMajor,\
              CblasNoTrans,\
              CblasNoTrans,\
              m,n,k,\
              1,\
              MatrixA->matrix,\
              k,\
              MatrixB->matrix,\
              n,\
              0,\
              result->matrix,\
              n);
  result->shape[0] = MatrixA->shape[0];
  result->shape[1] = MatrixB->shape[1];
  return result;
}

dARRAY * dotm(dARRAY * MatrixA, dARRAY * MatrixB){
  if(MatrixA->shape[1]!=MatrixB->shape[0]){
    printf("\033[1;31mError:\033[93m Shape error while performing dot(). Matrix dimensions do not align. %d(dim1) != %d(dim0)\033[0m\n",MatrixA->shape[1],MatrixB->shape[0]);
    return NULL;
  }
  if(MatrixB == NULL || MatrixA == NULL){
    printf("\033[1;31mError:\033[93m One of the input matrices is empty. Call dot() only after initializing dARRAY object\033[0m\n");
    return NULL;
  }
  dARRAY * BT = NULL;
  dARRAY * result = NULL;
  result = (dARRAY *)malloc(sizeof(dARRAY));
  result->matrix = (float*)calloc(MatrixA->shape[0]*MatrixB->shape[1],sizeof(float));
  BT = transpose(MatrixB);
  omp_set_num_threads(nn_threads);
  #pragma omp parallel for num_threads(nn_threads) collapse(1) schedule(static)
  for(int i=0;i<MatrixA->shape[0];i++){
    for(int j=0;j<MatrixB->shape[1];j++){
      for(int k=0;k<MatrixB->shape[0];k++){
        result->matrix[i * MatrixB->shape[1]+j] += MatrixA->matrix[i*MatrixA->shape[1]+k] * BT->matrix[j*MatrixB->shape[0]+k];
      }
    }
  }
  
  free2d(BT);
  BT = NULL;
  result->shape[0] = MatrixA->shape[0];
  result->shape[1] = MatrixB->shape[1];
  return result;
}

/**!
 * Function performs element-wise multiplication on two matrices. 
 * @param MatrixA First Matrix (float *) 
 * @param MatrixB Second Matrix (float *) 
 * @result Returns a pointer to the result of multiply(MatrixA,MatrixB) 
 * @return A pointer to the result of multiply(MatrixA,MatrixB) 
*/
dARRAY * multiply(dARRAY * restrict MatrixA, dARRAY * restrict MatrixB){
  if(MatrixB == NULL || MatrixA == NULL){
    printf("\033[1;31mError:\033[93m One of the input matrices is empty. Call multiply() only after initializing dARRAY object\033[0m\n");
    return NULL;
  }
  dARRAY * temp = NULL;
  int x = 0, y = 0;
  #pragma omp sections nowait
  {
    #pragma omp section
    x = size(MatrixA);
    #pragma omp section
    y = size(MatrixB);
  }  
  int flag = 0;
  if(x>y){ 
    temp = b_cast(MatrixA,MatrixB); 
    flag=1;
  }
  else if(x<y){
    temp = b_cast(MatrixB,MatrixA);
    flag=1;
  }
  if(temp==NULL && flag){
    printf("\033[1;31mError:\033[93m Could not perform multiply(). Please check shape of input matrices.\033[0m\n");
    return NULL;
  }
  //since both the matrices must have the same dimensions, we can use shape of any matrix
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (float*)calloc(MatrixA->shape[0]*MatrixA->shape[1],sizeof(float));
  if(x==y){
    omp_set_num_threads(nn_threads);
    int i = 0;
    int m = MatrixA->shape[0];
    int n = MatrixA->shape[1];
    float * matrixA, *matrixB,*res_matrix;
    matrixA = MatrixA->matrix;
    matrixB = MatrixB->matrix;
    res_matrix = result->matrix;
    #pragma omp parallel for num_threads(nn_threads) shared(matrixA,matrixB,res_matrix,m,n) private(i) schedule(static)
    for(i=0;i<m*n;i++)
      res_matrix[i] = matrixA[i] * matrixB[i];
  }
  else{
    omp_set_num_threads(nn_threads);
    int i = 0;
    int m = MatrixA->shape[0];
    int n = MatrixA->shape[1];
    float * matrixA, *matrixB,*res_matrix,*temp_matrix;
    matrixA = MatrixA->matrix;
    matrixB = MatrixB->matrix;
    temp_matrix = temp->matrix;
    res_matrix = result->matrix;
    #pragma omp parallel for num_threads(nn_threads) shared(matrixA,matrixB,res_matrix,temp_matrix,m,n,x,y) private(i) schedule(static)
    for(i=0;i<m*n;i++)
      res_matrix[i] = x>y ? matrixA[i] * temp_matrix[i] : temp_matrix[i] * matrixB[i];
  }
  if(temp!=NULL)
    free2d(temp);
  temp = NULL;
  result->shape[0] = MatrixA->shape[0];
  result->shape[1] = MatrixA->shape[1];
  return result;
}

/**!
 * Function performs element-wise divison on two matrices. 
 * @param MatrixA First Matrix (float *) 
 * @param MatrixB Second Matrix (float *) 
 * @result Returns a pointer to the result of divison(MatrixA,MatrixB) 
 * @return A pointer to the result of divison(MatrixA,MatrixB) 
*/
dARRAY * divison(dARRAY * restrict MatrixA, dARRAY * restrict MatrixB){
  if(MatrixB == NULL || MatrixA == NULL){
    printf("\033[1;31mError:\033[93m One of the input matrices is empty. Call divison() only after initializing dARRAY object\033[0m\n");
    return NULL;
  }
  dARRAY * temp = NULL;
  int x = size(MatrixA);
  int y = size(MatrixB);
  int flag=0;
  if(x>y){ 
    temp = b_cast(MatrixA,MatrixB);
    flag=1; 
  }
  else if(x<y){
    temp = b_cast(MatrixB,MatrixA);
    flag=1;
  }
  if(temp==NULL && flag){
    printf("\033[1;31mError:\033[93m Could not perform divison(). Please check shape of input matrices.\033[0m\n");
    return NULL;
  }
  //since both the matrices must have the same dimensions, we can use shape of any matrix
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (float*)calloc(MatrixA->shape[0]*MatrixA->shape[1],sizeof(float));
  if(x==y){
    omp_set_num_threads(nn_threads);
    int i = 0;
    int m = MatrixA->shape[0];
    int n = MatrixA->shape[1];
    float * matrixA, *matrixB,*res_matrix;
    matrixA = MatrixA->matrix;
    matrixB = MatrixB->matrix;
    res_matrix = result->matrix;
    #pragma omp parallel for num_threads(nn_threads) shared(matrixA,matrixB,res_matrix,m,n) private(i) schedule(static)
    for(i=0;i<m*n;i++){
      res_matrix[i] = matrixA[i] / matrixB[i];
    }
  }
  else{
    omp_set_num_threads(nn_threads);
    int i = 0;
    int m = MatrixA->shape[0];
    int n = MatrixA->shape[1];
    float * matrixA, *matrixB,*res_matrix,*temp_matrix;
    matrixA = MatrixA->matrix;
    matrixB = MatrixB->matrix;
    temp_matrix = temp->matrix;
    res_matrix = result->matrix;
    #pragma omp parallel for num_threads(nn_threads) shared(matrixA,matrixB,res_matrix,temp_matrix,m,n,x,y) private(i) schedule(static)
    for(i=0;i<m*n;i++)
        res_matrix[i] = x>y ? matrixA[i] / temp_matrix[i] : temp_matrix[i] / matrixB[i];
  }
  if(temp!=NULL)
    free2d(temp);
  temp = NULL;
  result->shape[0] = MatrixA->shape[0];
  result->shape[1] = MatrixA->shape[1];
  return result;
}

/**!
 * Function performs element-wise addition on two matrices. 
 * @param MatrixA First Matrix (float *) 
 * @param MatrixB Second Matrix (float *) 
 * @result Returns a pointer to the result of add(MatrixA,MatrixB) 
 * @return A pointer to the result of add(MatrixA,MatrixB) 
*/
dARRAY * add(dARRAY * MatrixA, dARRAY * MatrixB){
  if(MatrixB == NULL || MatrixA == NULL){
    printf("\033[1;31mError:\033[93m One of the input matrices is empty. Call add() only after initializing dARRAY object\033[0m\n");
    return NULL;
  }
  dARRAY * bcast_arr = NULL;
  int x = size(MatrixA);
  int y = size(MatrixB);
  int flag=0;
  if(x>y){ 
    bcast_arr = b_cast(MatrixA,MatrixB); 
    flag=1;
  }
  else if(x<y){
    bcast_arr = b_cast(MatrixB,MatrixA);
    flag=1;
  }
  if(bcast_arr==NULL && flag){
    printf("\033[1;31mError:\033[93m Could not perform add(). Please check shape of input matrices.\033[0m\n");
    return NULL;
  }
  //since both the matrices must have the same dimensions, we can use shape of any matrix
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (float*)calloc(MatrixA->shape[0]*MatrixA->shape[1],sizeof(float));
  if(x==y){
    // omp_set_num_threads(nn_threads);
    // #pragma omp parallel for num_threads(nn_threads) shared(MatrixA,MatrixB,result) schedule(static)
    // for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++){
    //   result->matrix[i] = MatrixA->matrix[i] + MatrixB->matrix[i];
    // }
    cblas_scopy(MatrixB->shape[0]*MatrixB->shape[1],MatrixB->matrix,1,result->matrix,1);
    cblas_saxpy(MatrixA->shape[0]*MatrixA->shape[1],1,MatrixA->matrix,1,result->matrix,1);
  }
  else{
    // omp_set_num_threads(nn_threads);
    // #pragma omp parallel for num_threads(nn_threads) shared(MatrixA,MatrixB,bcast_arr,result) schedule(static)
    // for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++)
    //     result->matrix[i] = x>y ? MatrixA->matrix[i] + bcast_arr->matrix[i] : bcast_arr->matrix[i] + MatrixB->matrix[i];
    if(x>y){
      cblas_scopy(MatrixA->shape[0]*MatrixA->shape[1],bcast_arr->matrix,1,result->matrix,1);
      cblas_saxpy(MatrixA->shape[0]*MatrixA->shape[1],1,MatrixA->matrix,1,result->matrix,1);
    }
    else{
      cblas_scopy(MatrixB->shape[0]*MatrixB->shape[1],MatrixB->matrix,1,result->matrix,1);
      cblas_saxpy(MatrixA->shape[0]*MatrixA->shape[1],1,bcast_arr->matrix,1,result->matrix,1);
    }
  }
  if(bcast_arr!=NULL)
  free2d(bcast_arr);
  result->shape[0] = MatrixA->shape[0];
  result->shape[1] = MatrixA->shape[1];
  return result;
}

/**!
 * Function performs element-wise subtraction on two matrices. 
 * @param MatrixA First Matrix (float *) 
 * @param MatrixB Second Matrix (float *) 
 * @result Returns a pointer to the result of subtract(MatrixA,MatrixB) 
 * @return A pointer to the result of subtract(MatrixA,MatrixB) 
*/
dARRAY * subtract(dARRAY * MatrixA, dARRAY * MatrixB){
  if(MatrixB == NULL || MatrixA == NULL){
    printf("\033[1;31mError:\033[93m One of the input matrices is empty. Call subtract() only after initializing dARRAY object\033[0m\n");
    return NULL;
  }
  dARRAY * bcast_arr = NULL;
  int x = size(MatrixA);
  int y = size(MatrixB);
  int flag=0;
  if(x>y){ 
    bcast_arr = b_cast(MatrixA,MatrixB);
  }
  else if(x<y){
    bcast_arr = b_cast(MatrixB,MatrixA);
    flag=1;
  }
  if(bcast_arr==NULL && flag==1){
    printf("\033[1;31mError:\033[93m Could not perform subtract(). Please check shape of input matrices.\033[0m\n");
    return NULL;
  }
  //since both the matrices must have the same dimensions, we can use shape of any matrix
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (float*)calloc(MatrixA->shape[0]*MatrixA->shape[1],sizeof(float));
  if(x==y){
    // omp_set_num_threads(nn_threads);
    // #pragma omp parallel for num_threads(nn_threads) shared(MatrixA,MatrixB,result) schedule(static)
    // for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++){
    //   result->matrix[i] = MatrixA->matrix[i] + MatrixB->matrix[i];
    // }
    cblas_scopy(MatrixA->shape[0]*MatrixA->shape[1],MatrixA->matrix,1,result->matrix,1);
    cblas_saxpy(MatrixB->shape[0]*MatrixB->shape[1],-1,MatrixB->matrix,1,result->matrix,1);
  }
  else{
    // omp_set_num_threads(nn_threads);
    // #pragma omp parallel for num_threads(nn_threads) shared(MatrixA,MatrixB,bcast_arr,result) schedule(static)
    // for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++)
    //     result->matrix[i] = x>y ? MatrixA->matrix[i] + bcast_arr->matrix[i] : bcast_arr->matrix[i] + MatrixB->matrix[i];
    if(x>y){
      cblas_scopy(MatrixA->shape[0]*MatrixA->shape[1],MatrixA->matrix,1,result->matrix,1);
      cblas_saxpy(MatrixA->shape[0]*MatrixA->shape[1],-1,bcast_arr->matrix,1,result->matrix,1);
    }
    else{
      cblas_scopy(bcast_arr->shape[0]*bcast_arr->shape[1],bcast_arr->matrix,1,result->matrix,1);
      cblas_saxpy(MatrixA->shape[0]*MatrixA->shape[1],-1,MatrixB->matrix,1,result->matrix,1);
    }
  }
  if(bcast_arr!=NULL)
    free2d(bcast_arr);
  bcast_arr = NULL;
  result->shape[0] = MatrixA->shape[0];
  result->shape[1] = MatrixA->shape[1];
  return result;
}

/**!
 * Function Adds a scalar value to each element of a matrix. 
 * @param matrix A matrix of dARRAY Object. 
 * @param scalar A scalar value that needs to be added to each element of matrix. 
 * @result A pointer to the result of addScalar(matrix,scalar) 
 * @return A pointer to the result of addScalar(matrix,scalar) 
*/
dARRAY * addScalar(dARRAY * matrix, float scalar){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Matrix is empty. Call addScalar() only after intializing dARRAY object.\033[0m\n");
    return NULL;
  }
  // float * mul_matrix = (float*)calloc(matrix->shape[0]*matrix->shape[1],sizeof(float));

  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (float*)calloc(matrix->shape[0]*matrix->shape[1],sizeof(float));
  omp_set_num_threads(nn_threads);
  #pragma omp parallel for num_threads(nn_threads) shared(matrix,result,scalar) schedule(static)
  for(int i=0; i<matrix->shape[0]*matrix->shape[1];  i++){
    result->matrix[i] = matrix->matrix[i] + scalar;
  }
  result->shape[0] = matrix->shape[0];
  result->shape[1] = matrix->shape[1];
  return result;
}

/**!
 * Function subtracts a scalar value from each element of a matrix. 
 * @param matrix A matrix of dARRAY Object. 
 * @param scalar A scalar value that needs to be subtracted from each element of matrix. 
 * @result A pointer to the result of subScalar(matrix,scalar) 
 * @return A pointer to the result of subScalar(matrix,scalar) 
*/
dARRAY * subScalar(dARRAY * matrix, float scalar){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Matrix is empty. Call subScalar() only after intializing dARRAY object.\033[0m\n");
    return NULL;
  }
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (float*)calloc(matrix->shape[0]*matrix->shape[1],sizeof(float));
  omp_set_num_threads(nn_threads);
  #pragma omp parallel for num_threads(nn_threads) shared(matrix,result,scalar) schedule(static)
  for(int i=0; i<matrix->shape[0]*matrix->shape[1]; i++){
    result->matrix[i] = matrix->matrix[i] - scalar;
  }
  result->shape[0] = matrix->shape[0];
  result->shape[1] = matrix->shape[1];
  return result;
}

/**!
 * Function multiplies a scalar value with each element of a matrix. 
 * @param matrix A matrix of dARRAY Object 
 * @param scalar A scalar value that needs to be multiplied with each element of matrix. 
 * @result A pointer to the result of mulScalar(matrix,scalar) 
 * @return A pointer to the result of mulScalar(matrix,scalar) 
*/
dARRAY * mulScalar(dARRAY * matrix, float scalar){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Matrix is empty. Call mulScalar() only after intializing dARRAY object.\033[0m\n");
    return NULL;
  }
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (float*)calloc(matrix->shape[0]*matrix->shape[1],sizeof(float));
  omp_set_num_threads(nn_threads);
  #pragma omp parallel for num_threads(nn_threads) shared(matrix,result,scalar) schedule(static)
  for(int i=0; i<matrix->shape[0]*matrix->shape[1]; i++){
    result->matrix[i] = matrix->matrix[i] * scalar;
  }
  result->shape[0] = matrix->shape[0];
  result->shape[1] = matrix->shape[1];
  return result;
}
dARRAY * mulScalarm(dARRAY * matrix, float scalar){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Matrix is empty. Call divScalar() only after intializing dARRAY object.\033[0m\n");
    return NULL;
  }
  float * div_mat = (float*)calloc(matrix->shape[0]*matrix->shape[1],sizeof(float));
  #pragma omp task
  cblas_sscal(matrix->shape[0]*matrix->shape[1],scalar,matrix->matrix,1);
  #pragma omp task
  cblas_scopy(matrix->shape[0]*matrix->shape[1],matrix->matrix,1,div_mat,1);
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = div_mat;
  result->shape[0] = matrix->shape[0];
  result->shape[1] = matrix->shape[1];
  return result;
}

/**!
 * Function divides a scalar value with each element of a matrix. 
 * @param matrix A matrix of dARRAY Object. 
 * @param scalar A scalar value that needs to be divided with each element of matrix. 
 * @result A pointer to the result of divScalar(matrix,scalar) 
 * @return A pointer to the result of divScalar(matrix,scalar) 
*/
dARRAY * divScalar(dARRAY * matrix, float scalar){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Matrix is empty. Call divScalar() only after intializing dARRAY object.\033[0m\n");
    return NULL;
  }
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (float*)calloc(matrix->shape[0]*matrix->shape[1],sizeof(float));
  omp_set_num_threads(nn_threads);
  #pragma omp parallel for num_threads(nn_threads) shared(matrix,result,scalar) schedule(static)
  for(int i=0; i<matrix->shape[0]*matrix->shape[1];  i++){
    result->matrix[i] = matrix->matrix[i] / scalar;
  }
  result->shape[0] = matrix->shape[0];
  result->shape[1] = matrix->shape[1];
  return result;
}

dARRAY * divScalarm(dARRAY * matrix, float scalar){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Matrix is empty. Call divScalar() only after intializing dARRAY object.\033[0m\n");
    return NULL;
  }
  float * div_mat = (float*)calloc(matrix->shape[0]*matrix->shape[1],sizeof(float));
  omp_set_num_threads(nn_threads);
  #pragma omp task
  cblas_sscal(matrix->shape[0]*matrix->shape[1],(1/scalar),matrix->matrix,1);
  #pragma omp task
  cblas_scopy(matrix->shape[0]*matrix->shape[1],matrix->matrix,1,div_mat,1);
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = div_mat;
  result->shape[0] = matrix->shape[0];
  result->shape[1] = matrix->shape[1];
  return result;
}

/**!
 * Function raises the elements of a matrix to the specified power. 
 * @param matrix A matrix of dARRAY Object 
 * @param power A value to which each element in matrix must be raised. 
 * @result A pointer to the result of power(matrix,power) 
 * @return A pointer to the result of power(matrix,power) 
*/
dARRAY * power(dARRAY * matrix, float power){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Matrix is empty. Call power() only after intializing dARRAY object.\033[0m\n");
    return NULL;
  }
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (float*)calloc(matrix->shape[0]*matrix->shape[1],sizeof(float));
  omp_set_num_threads(nn_threads);
  #pragma omp parallel for num_threads(nn_threads) shared(matrix,result,power) schedule(static)
  for(int i=0; i<matrix->shape[0]*matrix->shape[1];  i++){
    result->matrix[i] = (float)pow(matrix->matrix[i],power);
  }
  result->shape[0] = matrix->shape[0];
  result->shape[1] = matrix->shape[1];
  return result;
}

dARRAY * squareroot(dARRAY * matrix){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Matrix is empty. Call squareroot() only after intializing dARRAY object.\033[0m\n");
    return NULL;
  }
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (float*)calloc(matrix->shape[0]*matrix->shape[1],sizeof(float));
  omp_set_num_threads(nn_threads);
  #pragma omp parallel for num_threads(nn_threads) shared(matrix,result) schedule(static)
  for(int i=0; i<matrix->shape[0]*matrix->shape[1];  i++){
    result->matrix[i] = (float)sqrt(matrix->matrix[i]);
  }
  result->shape[0] = matrix->shape[0];
  result->shape[1] = matrix->shape[1];
  return result;
}

dARRAY * exponentional(dARRAY * matrix){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Matrix is empty. Call power() only after intializing dARRAY object.\033[0m\n");
    return NULL;
  }
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (float*)calloc(matrix->shape[0]*matrix->shape[1],sizeof(float));
  omp_set_num_threads(nn_threads);
  #pragma omp parallel for num_threads(nn_threads) shared(matrix,result) schedule(static)
  for(int i=0; i<matrix->shape[0]*matrix->shape[1]; i++){
    result->matrix[i] = exp(matrix->matrix[i]);
  }
  result->shape[0] = matrix->shape[0];
  result->shape[1] = matrix->shape[1];
  return result;
}

/**!
 * Function performs broadcasting of matrices 
 * Refer to www.numpy.org for detailed explanation of broadcasting. 
 * The implementation used here is similar to the one in www.numpy.org. 
 * @param MatrixA Matrix of dARRAY Object 
 * @param MatrixB Matrix of dARRAY Object 
 * @result A pointer to the broadcasted matrix 
 * @return A pointer to the broadcasted matrix 
*/
dARRAY * b_cast(dARRAY * MatrixA, dARRAY * MatrixB){
  dARRAY * b_castArr = NULL;
  
  if(MatrixA->shape[1]==MatrixB->shape[1] && MatrixB->shape[0]==1 && MatrixA->shape[0]>MatrixB->shape[0]){
    //B matrix has the shape of (1,n) 
    //we need to copy B m times
    //M(5,4) B(1,4)  repeat 5 * 4 = 20 times
    b_castArr = (dARRAY*)malloc(sizeof(dARRAY));
    b_castArr->matrix = (float*)calloc(MatrixA->shape[0]*MatrixA->shape[1],sizeof(float));
    float * bcast_matrix, *matrixB;
    bcast_matrix = b_castArr->matrix;
    matrixB = MatrixB->matrix;
    int m = MatrixA->shape[0];
    int n = MatrixB->shape[1];
    int i = 0;
    omp_set_num_threads(nn_threads);
    #pragma omp parallel for num_threads(nn_threads) shared(matrixB,bcast_matrix,m,n) private(i) schedule(static,8)
    for(i=0;i<m*n;i++){
      bcast_matrix[i] = matrixB[(i%n)];
    }
    b_castArr->shape[0] = MatrixA->shape[0];
    b_castArr->shape[1] = MatrixB->shape[1];
  }
  else if(MatrixA->shape[0]==MatrixB->shape[0] && MatrixB->shape[1]==1 && MatrixA->shape[1]>MatrixB->shape[1]){
    //B is of the form (m,1)
    //A is of (m,n)
    //copy column wise.
    b_castArr = (dARRAY*)malloc(sizeof(dARRAY));
    b_castArr->matrix = (float*)calloc(MatrixA->shape[0]*MatrixA->shape[1],sizeof(float));
    int k=0;
    float * bcast_matrix, *matrixB;
    bcast_matrix = b_castArr->matrix;
    matrixB = MatrixB->matrix;
    int m = MatrixA->shape[0];
    int n = MatrixA->shape[1];
    int i = 0;
    int j = 0;
    omp_set_num_threads(nn_threads);
    #pragma omp parallel for num_threads(nn_threads) shared(matrixB,bcast_matrix,m,n,k) private(i,j) schedule(static,8)
    for(i=0;i<m;i++){
      //copy b n times
      for(j=0;j<n;j++){
        bcast_matrix[k] = matrixB[i];
        k++;
      }
    }
    b_castArr->shape[0] = MatrixA->shape[0];
    b_castArr->shape[1] = MatrixA->shape[1];
  }
  return b_castArr;
}

/**!
 * Function raises the elements of a matrix to the specified power. 
 * @param matrix A matrix of dARRAY Object 
 * @param axis If axis == 1, then sums all elements in a row. If axis == 0, then sums all the elements in a column
 * @param dims An array of matrix dimensions [rows,columns] 
 * @result A pointer to the result of sum(matrix,axis) 
 * @return A pointer to the result of sum(matrix,axis) 
*/
dARRAY * sum(dARRAY * matrix, int axis){
  if(axis!=0 && axis!=1){
    printf("\033[1;31mError:\033[93m axis=%d not supported. Instead use axis=0 or axis=1\033[0m\n",axis);
    return NULL;
  }
  // if(matrix->shape[0]==1 || matrix->shape[1]==1) return matrix;
  dARRAY * new = (dARRAY*)malloc(sizeof(dARRAY));
  new->matrix = NULL;
  if(axis==0){
    new->matrix = (float*)calloc(matrix->shape[1],sizeof(float));
    dARRAY * temp = transpose(matrix);
    float sum_ = 0.0f;
    int i = 0;
    int j = 0;
    omp_set_num_threads(nn_threads);
    #pragma omp parallel for num_threads(nn_threads) shared(temp,new) private(i,j) reduction(+:sum_)
    for(i=0;i<temp->shape[0];i++){
      sum_=0.0;
      for(j=0;j<temp->shape[1];j++){
        sum_+= temp->matrix[i*temp->shape[1]+j];
      }
      new->matrix[i] = sum_;
    }
    new->shape[0] = 1;
    new->shape[1] = matrix->shape[1];
    free2d(temp);
    temp=NULL;
  }
  else if(axis==1){
      new->matrix = (float*)calloc(matrix->shape[0],sizeof(float));
      omp_set_num_threads(nn_threads);
      int j = 0, i = 0;
      float temp = 0.0f; 
      #pragma omp parallel for num_threads(nn_threads) shared(matrix,new) private(i,j) reduction(+:temp)
      for(i=0;i<matrix->shape[0];i++){
        temp = 0.0;
        for(j=0;j<matrix->shape[1];j++){
          temp += matrix->matrix[i*matrix->shape[1]+j];
        }
        new->matrix[i] = temp;
      }
    new->shape[0] = matrix->shape[0];
    new->shape[1] = 1;
  }
  return new;
}

float frobenius_norm(dARRAY * matrix){
  float frobenius_norm = 0.0;
  omp_set_num_threads(nn_threads);
  #pragma omp parallel for num_threads(nn_threads) shared(matrix) reduction(+:frobenius_norm) schedule(static)
  for(int i=0;i<matrix->shape[0]*matrix->shape[1];i++){
    frobenius_norm += pow(matrix->matrix[i],2);
  }
  return frobenius_norm;
}

float Manhattan_distance(dARRAY * matrix){
  float dist = 0.0;
  omp_set_num_threads(nn_threads);
  #pragma omp parallel for num_threads(nn_threads) shared(matrix) reduction(+:dist) schedule(static)
  for(int i=0;i<matrix->shape[0]*matrix->shape[1];i++){
    dist += abs(matrix->matrix[i]);
  }
  return dist;
}

/**!
 * Function generates a matrix of specified dimensions filled with random variables 
 * from normal distribution with mean 0 and unit standard deviation. 
 * @param dims An array of matrix dimensions [rows,columns] 
 * @result A pointer to the generated matrix. 
 * @return A pointer to the generated matrix. 
*/
dARRAY * randn(int * dims){
  dARRAY * matrix = (dARRAY*)malloc(sizeof(dARRAY));
  matrix->matrix = (float*)malloc(sizeof(float)*dims[0]*dims[1]);
  omp_set_num_threads(nn_threads);
  #pragma omp parallel for collapse(1) shared(matrix)
  for(int i=0;i<dims[0];i++){
    for(int j=0;j<dims[1];j++){
      matrix->matrix[i*dims[1]+j] = rand_norm(0.0,1.0);
    }
  }
  matrix->shape[0] = dims[0];
  matrix->shape[1] = dims[1];
  return matrix;
}

/**!
 * Function reshapes a given matrix to specified dimensions 
 * @param matrix Matrix to be reshaped 
 * @param dims An array of matrix dimension [rows,columns] 
 * @result Pointer to the reshaped matrix 
 * @return Pointer to the reshaped matrix 
*/
dARRAY * reshape(dARRAY * matrix, int * dims){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Matrix is empty. Call reshape() only after intializing dARRAY object.\033[0m\n");
    return NULL;
  }
  if(size(matrix)!=(dims[0]*dims[1])){
    printf("\033[1;31mError:\033[93m Shape Error. Matrix could not be reshaped to the specified dims.\033[0m\n");
    return matrix;
  }
  matrix->shape[0] = dims[0];
  matrix->shape[1] = dims[1];
  return matrix;
}

/**!
 * Function mean of a matrix. 
 * @param matrix A matrix of dARRAY Object 
 * @result Mean of a matrix 
 * @return Mean of a matrix 
*/
float mean(dARRAY * matrix){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Cannot find mean of empty matrix. Call mean() only after intializing dARRAY object.\033[0m\n");
    return (float)0;
  }
  float sum = 0;
  for(int i=0; i<matrix->shape[0]*matrix->shape[1];i++)
    sum += matrix->matrix[i];
  return sum/(matrix->shape[0]*matrix->shape[1]);
}

/**!
 * Function finds the variance of a matrix.
 * @param matrix A matrix of dARRAY Object 
 * @param type if type=='sample' then function finds the sample variance else it finds the population variance. 
 * @result Variance of the matrix 
 * @return Variance of the matrix 
*/
float var(dARRAY * matrix, char * type){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Cannot find variance of empty matrix. Call var() only after intializing dARRAY object.\033[0m\n");
    return (float)0;
  }
  float errorSum = 0;
  float xbar = mean(matrix);
  for(int i=0;i<matrix->shape[0]*matrix->shape[1];i++){
    errorSum += pow((matrix->matrix[i]-xbar),2);
  }
  if(!strcmp(type,(const char *)"sample"))
    return errorSum/(matrix->shape[0]*matrix->shape[1]-1);
  else if(!strcmp(type,(const char *)"population"))
    return errorSum/(matrix->shape[0]*matrix->shape[1]);
  else{
    printf("\033[1;31mError:\033[93m \"type\" parameter can only take values \"sample\" or \"population\".\033[0m\n");
    return (float)0;
  }
}

/**!
 * Function finds the standard deviation of matrix. 
 * @param matrix A matrix of dARRAY Object 
 * @param type if type=='sample' then function finds the sample std else it finds the population std. 
 * @result Standard deviation of matrix 
 * @return Standard deviation of matrix 
*/
float std(dARRAY * matrix, char * type){
  return pow(var(matrix,type), 0.5);
}

/**!
 * Helper function of gaussRandom() 
 * Function generates a random variable with normal distribution. 
 * @param cache A pointer to the cache value 
 * @param return_cache A pointer to check if cache has a value. 
 * @result A random variable of normal distribution. 
 * @return A random variable of normal distribution. 
*/
float gaussGenerator(float * cache, int * return_cache){
  if(*return_cache){
    *return_cache = 0;
    return *cache;
  }
  //use drand48 to generate random values from uniform distribution
  float u = 2.0 * drand48() - 1.0;
  float v = 2.0 * drand48() - 1.0;
  float r = u*u + v*v;
  if(r==0.0 || r>1) return gaussGenerator(cache,return_cache);
  float c = sqrt(-2*log(r)/r);
  *cache = c*v; //store this in cache
  *return_cache = 1;
  return u*c;
}

/**!
 * Function generates a random variable with normal distribution. 
 * @result A random variable of normal distribution. 
 * @return A random variable of normal distribution. 
*/ 
float gaussRandom(){
  cache=0.0;
  return_cache = 0;
  return gaussGenerator(&cache,&return_cache);
}

/**!
 * Function generates a random variable with normal distribution with specified mean and standard deviation. 
 * @param mu Mean 
 * @param std Standard Deviation 
 * @result A random variable of normal distribution [X ~ N(mu,std*std)]. 
 * @return A random variable of normal distribution [X ~ N(mu,std*std)]. 
*/
float rand_norm(float mu, float std){
  return mu+gaussRandom()*std;
}

/**!
 * Function deallocates a 2D Matrix. 
 * @param matrix Matrix that needs to be freed. 
 * @result void 
 * @return void 
*/
void free2d(dARRAY * matrix){
  if(matrix==NULL) {
    printf("\033[1;93mWarning:\033[93m Matrix is Empty. No need for deallocation.\033[0m\n");
    return;
  }
  free(matrix->matrix);
  free(matrix);
  // matrix = NULL;
  return;
}

/**!
 * Function returns the size of the matrix 
 * @param A Matrix of type dARRAY Object 
 * @result Total size of the matrix 
 * @return Total size of the matrix 
*/
int size(dARRAY * A){
  if(A==NULL){
    printf("\033[1;31mError:\033[93m Matrix is Empty. Call size() only after intializing dARRAY object.\033[0m\n");
    return 0;
  }
  return A->shape[0]*A->shape[1];
}

/**!
 * Function displays the shape of the matrix 
 * @param A Matrix of type dARRAY Object 
 * @result Prints the shape of input matrix 
 * @return void 
*/
void shape(dARRAY * A){
  if(A==NULL){
    printf("\033[1;31mError:\033[93m Matrix is Empty. Call shape() only after intializing dARRAY object.\033[0m\n");
    return;
  }
  //printf("first element of matrix is : %f\n",A->matrix[0]);
  printf("(%d,%d)\n",A->shape[0],A->shape[1]);
}

void sleep_my(int milliseconds) {
  //Function to create a time delay. Mimicks thread.sleep() of Java
  unsigned int duration = time(0) + (milliseconds/1000);
  while(time(0)<duration);
}

void cleanSTDIN() {
  //This function is used instead of fflush(stdin) as it is a bad practice to use it 
  //due to undefined behaviour.
  int ch;
  while ((ch = getchar()) != '\n' && ch != EOF){}
}

void get_safe_nn_threads(){
  int num_cpu_cores = sysconf(_SC_NPROCESSORS_CONF);
  if(num_cpu_cores<=16){
    nn_threads = num_cpu_cores*2;
  }
  else nn_threads = num_cpu_cores;
}