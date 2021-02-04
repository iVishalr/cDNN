#include "./utils.h"
#include <math.h>
double cache;
int return_cache;

/**!
 * Creates a matrix filled with zeros. 
 * @param dims An array of matrix dimensions (int)[rows,columns] 
 * @result A pointer to the created matrix. 
 * @return A pointer to the created matrix. 
*/
dARRAY * zeros(int * dims){
  dARRAY * matrix = (dARRAY*)malloc(sizeof(dARRAY));
  matrix->matrix = (double*)calloc(dims[0]*dims[1],sizeof(double));
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
  matrix->matrix = (double*)malloc(sizeof(double)*(dims[0]*dims[1]));
  omp_set_num_threads(4);
  #pragma omp parallel for 
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
  matrix->matrix = (double*)calloc((dims[0]*dims[1]),sizeof(double));
  omp_set_num_threads(4);
  #pragma omp parallel for collapse(1)
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
  matrix->matrix = (double*)calloc(Matrix->shape[0]*Matrix->shape[1],sizeof(double));
  omp_set_num_threads(8);
  #pragma omp parallel for num_threads(8) collapse(1) shared(Matrix,matrix) schedule(static)
  for(int i=0;i<Matrix->shape[0];i++)
    for(int j=0;j<Matrix->shape[1];j++)
      matrix->matrix[j*Matrix->shape[0]+i] = Matrix->matrix[i*Matrix->shape[1]+j];
  matrix->shape[0] = Matrix->shape[1];
  matrix->shape[1] = Matrix->shape[0];
  return matrix;
}

/**!
 * Finds the dot product (Matrix Multiplication) of two matrices. 
 * @param MatrixA First Matrix (double * __restrict__) 
 * @param MatrixB Second Matrix (double * __restrict__) 
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
  dARRAY * BT = NULL;
  dARRAY * result = NULL;
  result = (dARRAY *)malloc(sizeof(dARRAY));
  result->matrix = (double*)calloc(MatrixA->shape[0]*MatrixB->shape[1],sizeof(double));
  BT = transpose(MatrixB);
  omp_set_num_threads(8);
  #pragma omp parallel for num_threads(8) collapse(1) schedule(static)
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
 * @param MatrixA First Matrix (double *) 
 * @param MatrixB Second Matrix (double *) 
 * @result Returns a pointer to the result of multiply(MatrixA,MatrixB) 
 * @return A pointer to the result of multiply(MatrixA,MatrixB) 
*/
dARRAY * multiply(dARRAY * restrict MatrixA, dARRAY * restrict MatrixB){
  if(MatrixB == NULL || MatrixA == NULL){
    printf("\033[1;31mError:\033[93m One of the input matrices is empty. Call multiply() only after initializing dARRAY object\033[0m\n");
    return NULL;
  }
  dARRAY * temp = NULL;
  int x = size(MatrixA);
  int y = size(MatrixB);
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
  result->matrix = (double*)calloc(MatrixA->shape[0]*MatrixA->shape[1],sizeof(double));
  if(x==y){
    omp_set_num_threads(8);
    #pragma omp parallel for num_threads(8) shared(MatrixA,MatrixB,result) schedule(static)
    for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++){
      result->matrix[i] = MatrixA->matrix[i] * MatrixB->matrix[i];
    }
  }
  else{
    omp_set_num_threads(8);
    #pragma omp parallel for num_threads(8) shared(MatrixA,MatrixB,temp,result) schedule(static)
    for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++)
        result->matrix[i] = x>y ? MatrixA->matrix[i] * temp->matrix[i] : temp->matrix[i] * MatrixB->matrix[i];
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
 * @param MatrixA First Matrix (double *) 
 * @param MatrixB Second Matrix (double *) 
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
  result->matrix = (double*)calloc(MatrixA->shape[0]*MatrixA->shape[1],sizeof(double));
  if(x==y){
    omp_set_num_threads(8);
    #pragma omp parallel for num_threads(8) shared(MatrixA,MatrixB,result) schedule(static)
    for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++){
      result->matrix[i] = MatrixA->matrix[i] / MatrixB->matrix[i];
    }
  }
  else{
    omp_set_num_threads(8);
    #pragma omp parallel for num_threads(8) shared(MatrixA,MatrixB,temp,result) schedule(static)
    for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++)
        result->matrix[i] = x>y ? MatrixA->matrix[i] / temp->matrix[i] : temp->matrix[i] / MatrixB->matrix[i];
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
 * @param MatrixA First Matrix (double *) 
 * @param MatrixB Second Matrix (double *) 
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
  result->matrix = (double*)calloc(MatrixA->shape[0]*MatrixA->shape[1],sizeof(double));
  if(x==y){
    omp_set_num_threads(8);
    #pragma omp parallel for num_threads(8) shared(MatrixA,MatrixB,result) schedule(static)
    for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++){
      result->matrix[i] = MatrixA->matrix[i] + MatrixB->matrix[i];
    }
  }
  else{
    omp_set_num_threads(8);
    #pragma omp parallel for num_threads(8) shared(MatrixA,MatrixB,bcast_arr,result) schedule(static)
    for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++)
        result->matrix[i] = x>y ? MatrixA->matrix[i] + bcast_arr->matrix[i] : bcast_arr->matrix[i] + MatrixB->matrix[i];
  }
  if(bcast_arr!=NULL)
  free2d(bcast_arr);
  result->shape[0] = MatrixA->shape[0];
  result->shape[1] = MatrixA->shape[1];
  return result;
}

/**!
 * Function performs element-wise subtraction on two matrices. 
 * @param MatrixA First Matrix (double *) 
 * @param MatrixB Second Matrix (double *) 
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
    flag=1;
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
  result->matrix = (double*)calloc(MatrixA->shape[0]*MatrixA->shape[1],sizeof(double));
  if(x==y){
    omp_set_num_threads(8);
    #pragma omp parallel for num_threads(8) shared(MatrixA,MatrixB,result) schedule(static)
    for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++){
      result->matrix[i] = MatrixA->matrix[i] - MatrixB->matrix[i];
    }
  }
  else{
    omp_set_num_threads(8);
    #pragma omp parallel for num_threads(8) shared(MatrixA,MatrixB,bcast_arr,result) schedule(static)
    for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++)
        result->matrix[i] = x>y ? MatrixA->matrix[i] - bcast_arr->matrix[i] : bcast_arr->matrix[i] - MatrixB->matrix[i];
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
dARRAY * addScalar(dARRAY * matrix, double scalar){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Matrix is empty. Call addScalar() only after intializing dARRAY object.\033[0m\n");
    return NULL;
  }
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (double*)calloc(matrix->shape[0]*matrix->shape[1],sizeof(double));
  omp_set_num_threads(8);
  #pragma omp parallel for num_threads(8) shared(matrix,result,scalar) schedule(static)
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
dARRAY * subScalar(dARRAY * matrix, double scalar){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Matrix is empty. Call subScalar() only after intializing dARRAY object.\033[0m\n");
    return NULL;
  }
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (double*)calloc(matrix->shape[0]*matrix->shape[1],sizeof(double));
  omp_set_num_threads(8);
  #pragma omp parallel for num_threads(8) shared(matrix,result,scalar) schedule(static)
  for(int i=0; i<matrix->shape[0]*matrix->shape[1];  i++){
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
dARRAY * mulScalar(dARRAY * matrix, double scalar){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Matrix is empty. Call mulScalar() only after intializing dARRAY object.\033[0m\n");
    return NULL;
  }
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (double*)calloc(matrix->shape[0]*matrix->shape[1],sizeof(double));
  omp_set_num_threads(8);
  #pragma omp parallel for num_threads(8) shared(matrix,result,scalar) schedule(static)
  for(int i=0; i<matrix->shape[0]*matrix->shape[1];  i++){
    result->matrix[i] = matrix->matrix[i] * scalar;
  }
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
dARRAY * divScalar(dARRAY * matrix, double scalar){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Matrix is empty. Call divScalar() only after intializing dARRAY object.\033[0m\n");
    return NULL;
  }
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (double*)calloc(matrix->shape[0]*matrix->shape[1],sizeof(double));
  omp_set_num_threads(8);
  #pragma omp parallel for num_threads(8) shared(matrix,result,scalar) schedule(static)
  for(int i=0; i<matrix->shape[0]*matrix->shape[1];  i++){
    result->matrix[i] = matrix->matrix[i] / scalar;
  }
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
dARRAY * power(dARRAY * matrix, int power){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Matrix is empty. Call power() only after intializing dARRAY object.\033[0m\n");
    return NULL;
  }
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (double*)calloc(matrix->shape[0]*matrix->shape[1],sizeof(double));
  omp_set_num_threads(8);
  #pragma omp parallel for num_threads(8) shared(matrix,result,power) schedule(static)
  for(int i=0; i<matrix->shape[0]*matrix->shape[1];  i++){
    result->matrix[i] = pow(matrix->matrix[i],power);
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
    b_castArr->matrix = (double*)calloc(MatrixA->shape[0]*MatrixA->shape[1],sizeof(double));
    omp_set_num_threads(8);
    #pragma omp parallel for num_threads(8) shared(MatrixA,MatrixB,b_castArr) schedule(static)
    for(int i=0;i<MatrixA->shape[0]*MatrixB->shape[1];i++){
      b_castArr->matrix[i] = MatrixB->matrix[(i%MatrixB->shape[1])];
    }
    b_castArr->shape[0] = MatrixA->shape[0];
    b_castArr->shape[1] = MatrixB->shape[1];
  }
  else if(MatrixA->shape[0]==MatrixB->shape[0] && MatrixB->shape[1]==1 && MatrixA->shape[1]>MatrixB->shape[1]){
    //B is of the form (m,1)
    //A is of (m,n)
    //copy column wise.
    b_castArr = (dARRAY*)malloc(sizeof(dARRAY));
    b_castArr->matrix = (double*)calloc(MatrixA->shape[0]*MatrixA->shape[1],sizeof(double));
    int k=0;
    omp_set_num_threads(8);
    #pragma omp parallel for num_threads(8) shared(MatrixA,MatrixB,b_castArr,k) schedule(static)
    for(int i=0;i<MatrixA->shape[0];i++){
      //copy b n times
      for(int j=0;j<MatrixA->shape[1];j++){
        b_castArr->matrix[k] = MatrixB->matrix[i];
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
    new->matrix = (double*)calloc(matrix->shape[1],sizeof(double));
    omp_set_num_threads(8);
    #pragma omp parallel for num_threads(8) collapse(1) shared(matrix,new) schedule(static)
    for(int i = 0; i<matrix->shape[0];i++){
      double temp = 0.0;
      for(int j=0;j<matrix->shape[1];j++){
        temp += matrix->matrix[j*matrix->shape[1]+i];
      }
      new->matrix[i] = temp;
    }
    new->shape[0] = 1;
    new->shape[1] = matrix->shape[1];
  }
  else if(axis==1){
    new->matrix = (double*)calloc(matrix->shape[0],sizeof(double));
    omp_set_num_threads(8);
    #pragma omp parallel for num_threads(8) collapse(1) shared(matrix,new) schedule(static)
    for(int i=0;i<matrix->shape[0];i++){
      double temp = 0.0;
      for(int j=0;j<matrix->shape[1];j++){
        temp += matrix->matrix[i*matrix->shape[1]+j];
      }
      new->matrix[i] = temp;
    }
    new->shape[0] = matrix->shape[0];
    new->shape[1] = 1;
  }
  return new;
}

double frobenius_norm(dARRAY * matrix){
  double frobenius_norm = 0.0;
  omp_set_num_threads(8);
  #pragma omp parallel for num_threads(8) shared(matrix) schedule(static)
  for(int i=0;i<matrix->shape[0]*matrix->shape[1];i++){
    frobenius_norm += pow(matrix->matrix[i],2);
  }
  return frobenius_norm;
}

double Manhattan_distance(dARRAY * matrix){
  double dist = 0.0;
  omp_set_num_threads(8);
  #pragma omp parallel for num_threads(8) shared(matrix) schedule(static)
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
  matrix->matrix = (double*)malloc(sizeof(double)*dims[0]*dims[1]);
  omp_set_num_threads(8);
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
double mean(dARRAY * matrix){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Cannot find mean of empty matrix. Call mean() only after intializing dARRAY object.\033[0m\n");
    return (double)0;
  }
  double sum = 0;
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
double var(dARRAY * matrix, char * type){
  if(matrix==NULL){
    printf("\033[1;31mError:\033[93m Cannot find variance of empty matrix. Call var() only after intializing dARRAY object.\033[0m\n");
    return (double)0;
  }
  double errorSum = 0;
  double xbar = mean(matrix);
  for(int i=0;i<matrix->shape[0]*matrix->shape[1];i++){
    errorSum += pow((matrix->matrix[i]-xbar),2);
  }
  if(!strcmp(type,(const char *)"sample"))
    return errorSum/(matrix->shape[0]*matrix->shape[1]-1);
  else if(!strcmp(type,(const char *)"population"))
    return errorSum/(matrix->shape[0]*matrix->shape[1]);
  else{
    printf("\033[1;31mError:\033[93m \"type\" parameter can only take values \"sample\" or \"population\".\033[0m\n");
    return (double)0;
  }
}

/**!
 * Function finds the standard deviation of matrix. 
 * @param matrix A matrix of dARRAY Object 
 * @param type if type=='sample' then function finds the sample std else it finds the population std. 
 * @result Standard deviation of matrix 
 * @return Standard deviation of matrix 
*/
double std(dARRAY * matrix, char * type){
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
double gaussGenerator(double * cache, int * return_cache){
  if(*return_cache){
    *return_cache = 0;
    return *cache;
  }
  //use drand48 to generate random values from uniform distribution
  double u = 2.0 * drand48() - 1.0;
  double v = 2.0 * drand48() - 1.0;
  double r = u*u + v*v;
  if(r==0.0 || r>1) return gaussGenerator(cache,return_cache);
  double c = sqrt(-2*log(r)/r);
  *cache = c*v; //store this in cache
  *return_cache = 1;
  return u*c;
}

/**!
 * Function generates a random variable with normal distribution. 
 * @result A random variable of normal distribution. 
 * @return A random variable of normal distribution. 
*/ 
double gaussRandom(){
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
double rand_norm(double mu, double std){
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
  //printf("first element of matrix is : %lf\n",A->matrix[0]);
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