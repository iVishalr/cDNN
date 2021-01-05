#include "./utils.h"

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

  omp_set_num_threads(4);
  dARRAY * matrix = (dARRAY*)malloc(sizeof(dARRAY));
  matrix->matrix = (double *)malloc(sizeof(double)*Matrix->shape[0]*Matrix->shape[1]);
  #pragma omp parallel for shared(Matrix,matrix)
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
dARRAY * dot(dARRAY * restrict MatrixA, dARRAY * restrict MatrixB){
  if(MatrixA->shape[1]!=MatrixB->shape[0]){
    printf("\033[1;31mError:\033[93m Shape error while performing dot(). Matrix dimensions do not align. %d(dim1) != %d(dim0)\033[0m\n",MatrixA->shape[1],MatrixB->shape[0]);
    return NULL;
  }
  if(MatrixB == NULL || MatrixA == NULL){
    printf("\033[1;31mError:\033[93m One of the input matrices is empty. Call dot() only after initializing dARRAY object\033[0m\n");
    return NULL;
  }
  dARRAY * BT = NULL;
  double * res = (double *)malloc(sizeof(double)*MatrixA->shape[0]*MatrixB->shape[1]);
  BT = transpose(MatrixB);
  omp_set_num_threads(4);
  #pragma omp parallel for collapse(1) shared(MatrixA,MatrixB,res)
  for(int i=0;i<MatrixA->shape[0];i++){
    for(int j=0;j<MatrixB->shape[1];j++){
      for(int k=0;k<MatrixB->shape[0];k++){
        res[i * MatrixB->shape[1]+j] += MatrixA->matrix[i*MatrixA->shape[1]+k] * BT->matrix[j*MatrixB->shape[0]+k];
      }
    }
  }
  free2d(BT);
  BT = NULL;
  dARRAY * result = NULL;
  result = (dARRAY *)malloc(sizeof(dARRAY));
  result->matrix = res;
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
  result->matrix = (double*)malloc(sizeof(double)*MatrixA->shape[0]*MatrixA->shape[1]);
  if(x==y){
    for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++){
      result->matrix[i] = MatrixA->matrix[i] * MatrixB->matrix[i];
    }
  }
  else{
    for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++)
        result->matrix[i] = x>y ? MatrixA->matrix[i] * temp->matrix[i] : temp->matrix[i] * MatrixB->matrix[i];
  }
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
  result->matrix = (double*)malloc(sizeof(double)*MatrixA->shape[0]*MatrixA->shape[1]);
  if(x==y){
    for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++){
      result->matrix[i] = MatrixA->matrix[i] / MatrixB->matrix[i];
    }
  }
  else{
    for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++)
        result->matrix[i] = x>y ? MatrixA->matrix[i] / temp->matrix[i] : temp->matrix[i] / MatrixB->matrix[i];
  }
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
    printf("\033[1;31mError:\033[93m Could not perform add(). Please check shape of input matrices.\033[0m\n");
    return NULL;
  }
  //since both the matrices must have the same dimensions, we can use shape of any matrix
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (double*)malloc(sizeof(double)*MatrixA->shape[0]*MatrixA->shape[1]);
  if(x==y){
    for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++){
      result->matrix[i] = MatrixA->matrix[i] + MatrixB->matrix[i];
    }
  }
  else{
    for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++)
        result->matrix[i] = x>y ? MatrixA->matrix[i] + temp->matrix[i] : temp->matrix[i] + MatrixB->matrix[i];
  }
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
  if(temp==NULL && flag==1){
    printf("\033[1;31mError:\033[93m Could not perform subtract(). Please check shape of input matrices.\033[0m\n");
    return NULL;
  }
  //since both the matrices must have the same dimensions, we can use shape of any matrix
  dARRAY * result = (dARRAY*)malloc(sizeof(dARRAY));
  result->matrix = (double*)malloc(sizeof(double)*MatrixA->shape[0]*MatrixA->shape[1]);
  if(x==y){
    for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++){
      result->matrix[i] = MatrixA->matrix[i] - MatrixB->matrix[i];
    }
  }
  else{
    for(int i=0;i<MatrixA->shape[0]*MatrixA->shape[1];i++)
        result->matrix[i] = x>y ? MatrixA->matrix[i] - temp->matrix[i] : temp->matrix[i] - MatrixB->matrix[i];
  }
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
  for(int i=0; i<matrix->shape[0]*matrix->shape[1];  i++){
    matrix->matrix[i] = matrix->matrix[i] + scalar;
  }
  return matrix;
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
  for(int i=0; i<matrix->shape[0]*matrix->shape[1]; i++){
    matrix->matrix[i] = matrix->matrix[i] - scalar;
  }
  return matrix;
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
  for(int i=0;i<matrix->shape[0]*matrix->shape[1];i++){
    matrix->matrix[i] = matrix->matrix[i] * scalar;
  }
  return matrix;
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
  for(int i=0;i<matrix->shape[0]*matrix->shape[1];i++){
    matrix->matrix[i] = matrix->matrix[i] / scalar;
  }
  return matrix;
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
  for(int i=0;i<matrix->shape[0]*matrix->shape[1];i++){
    matrix->matrix[i] = pow(matrix->matrix[i],power);
  }
  return matrix;
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
    b_castArr->matrix = (double*)malloc(sizeof(double)*MatrixA->shape[0]*MatrixB->shape[1]);
    for(int i=0;i<MatrixA->shape[0]*MatrixB->shape[1];i++){
      b_castArr->matrix[i] = MatrixB->matrix[(i%MatrixB->shape[1])];
    }
    b_castArr->shape[0] = MatrixA->shape[0];
    b_castArr->shape[1] = MatrixB->shape[1];
  }
  else if(MatrixA->shape[0]==MatrixB->shape[0] && MatrixB->shape[1]==1 && MatrixA->shape[1]>MatrixB->shape[1]){
    //B is of the form (m,1)
    //A is of (m,n)
    //copy column wise
    b_castArr = (dARRAY*)malloc(sizeof(dARRAY));
    b_castArr->matrix = (double*)malloc(sizeof(double)*MatrixA->shape[0]*MatrixA->shape[1]);
    int k=0;
    #pragma omp parallel for shared(MatrixA,MatrixB,k)
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
  // else{
  //   printf("\033[1;31mError:\033[93m Matrices of shape (%d,%d) and (%d,%d) could not be broadcasted! Please input matrices with broadcastable dims.\n\033[1;36m(Refer www.numpy.org for more information on broadcasting)\033[0m\n",MatrixA->shape[0],MatrixA->shape[1],MatrixB->shape[0],MatrixB->shape[1]);
  //   return NULL;
  // }
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
  if(axis!=0 || axis!=1){
    printf("\033[1;31mError:\033[93m axis=%d not supported. Instead use axis=0 or axis=1\033[0m\n",axis);
    return NULL;
  }
  dARRAY * new = (dARRAY*)malloc(sizeof(dARRAY));
  double * res = NULL;
  if(axis==0){
    res = (double*)malloc(sizeof(double)*matrix->shape[1]);
    for(int i = 0; i<matrix->shape[0];i++){
      double temp = 0.0;
      for(int j=0;j<matrix->shape[1];j++){
        temp += matrix->matrix[j*matrix->shape[1]+i];
      }
      res[i] = temp;
    }
    new->shape[0] = 1;
    new->shape[1] = matrix->shape[1];
  }
  else if(axis==1){
    res = (double*)malloc(sizeof(double)*matrix->shape[0]);
    for(int i=0;i<matrix->shape[0];i++){
      double temp = 0.0;
      for(int j=0;j<matrix->shape[1];j++){
        temp += matrix->matrix[i*matrix->shape[1]+j];
      }
      res[i] = temp;
    }
    new->shape[0] = matrix->shape[0];
    new->shape[1] = 1;
  }
  new->matrix = res;
  return new;
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
  omp_set_num_threads(4);
  #pragma omp parallel for shared(matrix)
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
    printf("\033[1;31mError:\033[93m Matrix is Empty. No need for deallocation.\033[0m\n");
    return;
  }
  free(matrix->matrix);
  free(matrix);
  matrix = NULL;
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
  printf("(%d,%d)\n",A->shape[0],A->shape[1]);
}