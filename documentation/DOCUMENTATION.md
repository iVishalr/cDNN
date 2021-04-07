#<center>cDNN</center>

---

**cDNN** is a Deep Learning Library written in C Programming Language.

## Documentation

cDNN provides functions that can be used to create Artificial Neural Networks (ANN). These functions are designed to be as efficient as possible both in performance and memory.

#### Matrix Datatype

This section deals with creation of matrices in cDNN and organization of the matrix in memory.

```C
typedef struct array{
  float * matrix;
  int shape[2];
}dARRAY;
```

The above structure is used to create matrices in cDNN. `float * matrix` stores the elements of the matrix and `int shape[2]` stores the shape or the order of the matrix.

```C
int Matrix_dims = {2,2};
dARRAY * Matrix = zeros(Matrix_dims);
```

The above example creates a `(2,2)` matrix where each element is zero. `zeros()` allocates memory to the `dARRAY` matrix and returns it. `dARRAY * Matrix` is pointing to the matrix created by `zeros()`.

##### Organization of Matrix Elements in Memory

The elements of a matrix are stored in a RowMajor fashion in memory.

Consider the following matrix :
$$matrix = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}$$

The matrix has a shape `(3,3)`. The elements of the matrix would be stored in memory as follows :
$$matrix = \begin{bmatrix} 1. & 2. & 3. & 4. & 5. & 6. & 7. & 8. & 9. \end{bmatrix}$$

`float * matrix` stores the above array and `int shape[2] = {3,3}`. The shape of the matrix helps us to know the dimensions of the matrix.

We can also assign matrices to `dARRAY`

```C
dARRAY * Matrix = (dARRAY*)malloc(sizof(dARRAY));
Matrix->matrix = matrix;
Matrix->shape[0] = 3;
Matrix->shape[1] = 3;
```

Here `matrix` should be a pointer to array of floats.

Usually user need not use this way of assigning matrices. The matrix library that cDNN comes with does this by default. We will go in detail about the matrix library in the following section. For now, a small example can be used to explain it.

```C
int dims[] = {5,4};

//creating matrices A and B of shapes (5,4)
dARRAY * A = randn(dims);
dARRAY * B = ones(dims);

//Performing matrix-matrix operations on A and B
dARRAY * C = add(A,B);

free2d(A);
free2d(B);
```

In the above example, matrices A and B were created using `randn() & ones()` functions from matrix library of cDNN. The functions allocate memory for the matrix and return a pointer to the matrix.

Therefore, A and B will be pointing to a matrix (linear array) in memory. A and B will also have the appropriate shapes which will be stored in the `int shape[2]` field of `dARRAY`.

The matrix-matrix operations takes in matrices and applies an operation and returns a pointer to the result of operation. It is user's responsibility to free the input matrices if they are no longer used.

The main advantage of this type of matrix organization is that it eliminates the use of `float ** matrix` to store a 2D matrix. Operations would be very slow if we used `float ** matrix` due to double lookup table for pointers in memory.

Access of elements of a matrix can be done using the following way :

```C
int dims[] = {3,3};
dARRAY * A = ones(dims); //creates a (3,3) matrix of ones
...
//printing elements of matrix
for(i=0;i<A->shape[0];i++)
  for(j=0;j<A->shape[1];j++)
    printf("%f",A->matrix[i*A->shape[1]+j]);
```

`A->matrix[i*A->shape[1]+j]` allows us to access each element in the matrix.

#### 1. Matrix Operations

In the last section we looked briefly on one or two matrix functions. In this section, we will go much deeper into the basic operations of each of the matrix operations in cDNN matrix library.

##### 1. 1. `zeros()`

Creates a matrix a zero elements with the specified dimensions.

_Prototype_ :

```C
dARRAY * zeros(int * dims)
```

_Example_ :

```C
int dims = {10,10};
dARRAY * A = zeros(dims);
```

##### 1. 2. `ones()`

Creates a matrix with all elements equal to 1. The matrix created will be of the specified dimensions.

_Prototype_ :

```C
dARRAY * ones(int * dims)
```

_Example_ :

```C
int dims = {1,10};
dARRAY * A = ones(dims);
```

##### 1. 3. `eye()`

Creates an identity matrix with the specified dimensions.

_Prototype_ :

```C
dARRAY * eye(int * dims)
```

_Example_ :

```C
int dims = {3,3};
dARRAY * A = eye(dims);
```

##### 1. 4. `randn()`

Creates a matrix whose elements are random numbers from a standard normal distribution. The matrix created is of specified dimensions.

_Prototype_ :

```C
dARRAY * randn(int * dims)
```

_Example_ :

```C
int dims = {8,5};
dARRAY * A = randn(dims);
```

##### 1. 5. `reshape()`

Reshapes a given matrix with the specified dimensions.

**Note** : The reshaped matrix must have the same number of elements as the original matrix. Otherwise there will be a `shape error`.

_Prototype_ :

```C
dARRAY * reshape(dARRAY * matrix, int * dims)
```

_Example_ :

```C
int dims = {3,3};
dARRAY * A = eye(dims);

int new_dims = {1,9};
dARRAY * B = reshape(A,new_dims);
```

_Output_ :
$Input : A = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $Output : B = \begin{bmatrix} 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 \end{bmatrix}$

##### 1. 6. `size()`

Returns the number of elements in the matrix.

_Prototype_ :

```C
int size(dARRAY * matrix)
```

_Example_ :

```C
int dims = {3,3};
dARRAY * A = eye(dims);

printf("%d",size(A));
```

_Output_ :
$9$

##### 1. 7. `add()`

Adds two input matrices element wise and stores the result into another matrix and returns the pointer to the resultant matrix.

If two matrices are not of the same shape, the matrix that is smaller in shape would be broadcasted to match the shape of the bigger matrix. If the matrices are not broadcastable, an error will be thrown.

_Prototype_ :

```C
dARRAY * add(dARRAY * MatrixA, dARRAY * MatrixB)
```

_Example_ :

```C
int dims = {3,3};
dARRAY * A = ones(dims);
dARRAY * B = ones(dims);

dARRAY * C = add(A,B);
```

_Output_ :
$C = \begin{bmatrix} 2 & 2 & 2 \\ 2 & 2 & 2 \\ 2 & 2 & 2 \end{bmatrix}$

_Example_ :

```C
int A_dims = {3,3};
dARRAY * A = ones(A_dims);

int B_dims = {1,3};
dARRAY * B = ones(B_dims);

dARRAY * C = add(A,B);
```

_Output_ :
$Input : A = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}_{(3,3)} , B = \begin{bmatrix} 1 & 1 & 1\end{bmatrix}_{(1,3)}$

$broadcast(B = \begin{bmatrix} 1 & 1 & 1\end{bmatrix}_{(1,3)}) = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}_{(3,3)}$

$Output : C =  \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}_A +  \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}_B = \begin{bmatrix} 2 & 2 & 2 \\ 2 & 2 & 2 \\ 2 & 2 & 2 \end{bmatrix}$

_Example_ :

```C
int A_dims = {4,4};
dARRAY * A = ones(A_dims);

int B_dims = {4,1};
dARRAY * B = ones(B_dims);

dARRAY * C = add(A,B);
```

_Output_ :
$Input : A = \begin{bmatrix} 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\end{bmatrix}_{(4,4)} , B = \begin{bmatrix} 1 \\ 1 \\ 1 \\ 1\end{bmatrix}_{(4,1)}$

$broadcast(B = \begin{bmatrix} 1\\ 1\\ 1\\ 1\end{bmatrix}_{(4,1)}) = \begin{bmatrix} 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\end{bmatrix}_{(4,4)}$

$Output : C =  \begin{bmatrix} 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\end{bmatrix}_A +  \begin{bmatrix} 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\end{bmatrix}_B = \begin{bmatrix} 2 & 2 & 2 & 2\\ 2 & 2 & 2 & 2\\ 2 & 2 & 2 & 2\\2 & 2 & 2 & 2\end{bmatrix}$

##### 1. 8. `subtract()`

Subtracts two input matrices element wise and stores the result into another matrix and returns the pointer to the resultant matrix.

**Note** : Subtracts second matrix from first matrix and stores result in another matrix.

If two matrices are not of the same shape, the matrix that is smaller in shape would be broadcasted to match the shape of the bigger matrix. If the matrices are not broadcastable, an error will be thrown.

_Prototype_ :

```C
dARRAY * subtract(dARRAY * MatrixA, dARRAY * MatrixB)
```

_Example_ :

```C
int A_dims = {4,4};
dARRAY * A = ones(A_dims);

int B_dims = {4,1};
dARRAY * B = ones(B_dims);

dARRAY * C = subtract(A,B);
```

_Output_ :
$Input : A = \begin{bmatrix} 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\end{bmatrix}_{(4,4)} , B = \begin{bmatrix} 1 \\ 1 \\ 1 \\ 1\end{bmatrix}_{(4,1)}$

$broadcast(B = \begin{bmatrix} 1\\ 1\\ 1\\ 1\end{bmatrix}_{(4,1)}) = \begin{bmatrix} 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\end{bmatrix}_{(4,4)}$

$Output : C =  \begin{bmatrix} 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\end{bmatrix}_A -  \begin{bmatrix} 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\\ 1 & 1 & 1 & 1\end{bmatrix}_B = \begin{bmatrix} 0 & 0 & 0 & 0\\ 0 & 0 & 0 & 0\\ 0 & 0 & 0 & 0\\0 & 0 & 0 & 0\end{bmatrix}$

##### 1. 9. `multiply()`

Multiplies two input matrices element wise and stores the result into another matrix and returns the pointer to the resultant matrix.

If two matrices are not of the same shape, the matrix that is smaller in shape would be broadcasted to match the shape of the bigger matrix. If the matrices are not broadcastable, an error will be thrown.

_Prototype_ :

```C
dARRAY * multiply(dARRAY * MatrixA, dARRAY * MatrixB)
```

_Example_ :

```C
int A_dims = {2,2};
dARRAY * A = randn(A_dims);

int B_dims = {2,1};
dARRAY * B = randn(B_dims);

dARRAY * C = multiply(A,B);
```

_Output_ :
$Input : A = \begin{bmatrix} 0.432 & 0.213\\ 0.8453 & 0.4331\end{bmatrix}_{(2,2)} , B = \begin{bmatrix} 0.5323 \\ 0.2134\end{bmatrix}_{(2,1)}$

$broadcast(B = \begin{bmatrix} 0.5323 \\ 0.2134\end{bmatrix}_{(2,1)}) = B = \begin{bmatrix} 0.5323 & 0.5323\\ 0.2134 & 0.2134\end{bmatrix}_{(2,2)}$

$Output : C =  \begin{bmatrix} 0.432 & 0.213\\ 0.8453 & 0.4331\end{bmatrix}_A* \begin{bmatrix} 0.5323 & 0.5323\\ 0.2134 & 0.2134\end{bmatrix}_B = \begin{bmatrix} 0.2299 & 0.1133 \\ 0.1803 & 0.0924\end{bmatrix}$

##### 1. 10. `divison()`

Divides two input matrices element wise and stores the result into another matrix and returns the pointer to the resultant matrix.

**Note** : Divides second matrix from first matrix and stores result in another matrix.

If two matrices are not of the same shape, the matrix that is smaller in shape would be broadcasted to match the shape of the bigger matrix. If the matrices are not broadcastable, an error will be thrown.

_Prototype_ :

```C
dARRAY * divison(dARRAY * MatrixA, dARRAY * MatrixB)
```

_Example_ :

```C
int A_dims = {2,2};
dARRAY * A = randn(A_dims);

int B_dims = {2,1};
dARRAY * B = randn(B_dims);

dARRAY * C = divison(A,B);
```

_Output_ :
$Input : A = \begin{bmatrix} 0.432 & 0.213\\ 0.8453 & 0.4331\end{bmatrix}_{(2,2)} , B = \begin{bmatrix} 0.5323 \\ 0.2134\end{bmatrix}_{(2,1)}$

$broadcast(B = \begin{bmatrix} 0.5323 \\ 0.2134\end{bmatrix}_{(2,1)}) = B = \begin{bmatrix} 0.5323 & 0.5323\\ 0.2134 & 0.2134\end{bmatrix}_{(2,2)}$

$Output : C =  \begin{bmatrix} 0.432 & 0.213\\ 0.8453 & 0.4331\end{bmatrix} ÷ \begin{bmatrix} 0.5323 & 0.5323\\ 0.2134 & 0.2134\end{bmatrix}_B = \begin{bmatrix} 0.8115 & 0.4001 \\ 3.9611 & 2.0295\end{bmatrix}$

##### 1. 11. `addScalar()`

Adds a scalar to a matrix element wise and stores the result into another matrix and returns the pointer to the resultant matrix.

_Prototype_ :

```C
dARRAY * addScalar(dARRAY * matrix, float scalar)
```

_Example_ :

```C
int A_dims = {2,2};
dARRAY * A = ones(A_dims);

dARRAY * B = addScalar(A,10.0);
```

_Output_ :
$Input : A = \begin{bmatrix} 1 & 1\\ 1 & 1\end{bmatrix}_{(2,2)}$

$Output : B =  \begin{bmatrix} 1 & 1\\ 1 & 1\end{bmatrix} + 10.0 = \begin{bmatrix} 11.0 & 11.0 \\ 11.0 & 11.0\end{bmatrix}$

##### 1. 12. `subScalar()`

Subtracts a scalar from a matrix element wise and stores the result into another matrix and returns the pointer to the resultant matrix.

_Prototype_ :

```C
dARRAY * subScalar(dARRAY * matrix, float scalar)
```

_Example_ :

```C
int A_dims = {2,2};
dARRAY * A = ones(A_dims);

dARRAY * B = subScalar(A,10.0);
```

_Output_ :
$Input : A = \begin{bmatrix} 1 & 1\\ 1 & 1\end{bmatrix}_{(2,2)}$

$Output : B =  \begin{bmatrix} 1 & 1\\ 1 & 1\end{bmatrix} - 10.0 = \begin{bmatrix} -9.0 & -9.0 \\ -9.0 & -9.0\end{bmatrix}$

##### 1. 13. `mulScalar()`

Multiplies a scalar with a matrix element wise and stores the result into another matrix and returns the pointer to the resultant matrix.

_Prototype_ :

```C
dARRAY * mulScalar(dARRAY * matrix, float scalar)
```

_Example_ :

```C
int A_dims = {2,2};
dARRAY * A = ones(A_dims);

dARRAY * B = mulScalar(A,10.0);
```

_Output_ :
$Input : A = \begin{bmatrix} 1 & 1\\ 1 & 1\end{bmatrix}_{(2,2)}$

$Output : B =  \begin{bmatrix} 1 & 1\\ 1 & 1\end{bmatrix} * 10.0 = \begin{bmatrix} 10.0 & 10.0 \\ 10.0 & 10.0\end{bmatrix}$

##### 1. 14. `divScalar()`

Divides a scalar from a matrix element wise and stores the result into another matrix and returns the pointer to the resultant matrix.

_Prototype_ :

```C
dARRAY * divScalar(dARRAY * matrix, float scalar)
```

_Example_ :

```C
int A_dims = {2,2};
dARRAY * A = ones(A_dims);

dARRAY * B = divScalar(A,10.0);
```

_Output_ :
$Input : A = \begin{bmatrix} 1 & 1\\ 1 & 1\end{bmatrix}_{(2,2)}$

$Output : B =  \begin{bmatrix} 1 & 1\\ 1 & 1\end{bmatrix} ÷ 10.0 = \begin{bmatrix} 0.1 & 0.1 \\ 0.1 & 0.1\end{bmatrix}$

##### 1. 15. `power()`

Raises each element of matrix to a specfied power and stores the result into another matrix and returns the pointer to the resultant matrix.

_Prototype_ :

```C
dARRAY * power(dARRAY * matrix, float power)
```

_Example_ :

```C
int A_dims = {2,2};
dARRAY * A = ones(A_dims);

dARRAY * B = power(A,2.0);
```

##### 1. 16. `squareroot()`

Performs square root of each element of matrix and stores the result into another matrix and returns the pointer to the resultant matrix.

_Prototype_ :

```C
dARRAY * squareroot(dARRAY * matrix)
```

_Example_ :

```C
int A_dims = {2,2};
dARRAY * A = ones(A_dims);

dARRAY * B = squareroot(A);
```

##### 1. 17. `exponential()`

Performs `exp()` on each element of matrix and stores the result into another matrix and returns the pointer to the resultant matrix.

_Prototype_ :

```C
dARRAY * exponential(dARRAY * matrix)
```

_Example_ :

```C
int A_dims = {2,2};
dARRAY * A = ones(A_dims);

dARRAY * B = exponential(A);
```

##### 1. 18. `dot()`

Performs matrix multiplication of two input matrices and stores the result into another matrix and returns the pointer to the resultant matrix.

This must follow rules of matrix multiplication. Number of columns in first matrix must be equal to number of rows in second matrix.

_Prototype_ :

```C
dARRAY * dot(dARRAY * MatrixA, dARRAY * MatrixB)
```

_Example_ :

```C
int A_dims = {2,2};
dARRAY * A = randn(A_dims);

int B_dims = {2,4};
dARRAY * B = randn(B_dims);

dARRAY * C = dot(A,B);

//dot(B,A) gives shape error
```

This function uses a subroutine from `cblas.h` called `cblas_sgemm()`

Internal Implementation :

```C
cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,\
              m,n,k,\
              1,\
              MatrixA->matrix,\
              k,\
              MatrixB->matrix,\
              n,\
              0,\
              result->matrix,\
              n);
```

##### 1. 19. `transpose()`

Performs matrix transpose and stores the result into another matrix and returns the pointer to the resultant matrix.

_Prototype_ :

```C
dARRAY * transpose(dARRAY * matrix)
```

_Example_ :

```C
int dims = {2,2};
dARRAY * A = randn(dims);

dARRAY * C = transpose(A);
```

This function uses a subroutine from `cblas.h` called `cblas_somatcopy()`

Internal Implementation :

```C
cblas_somatcopy(CblasRowMajor,CblasTrans,\
                Matrix->shape[0],Matrix->shape[1],\
                1,Matrix->matrix,Matrix->shape[1],\
                matrix->matrix,Matrix->shape[0]);
```

##### 1. 20. `sum()`

Sum elements either along rows or columns depending on the axis specified and stores the result into another matrix and returns the pointer to the resultant matrix.

If `axis=0` then `sum()` performs sum of elements of matrix across columns. Else if `axis=1` then `sum()` performs sum of elements of matrix across rows.
_Prototype_ :

```C
dARRAY * sum(dARRAY * matrix, int axis)
```

_Example_ :

```C
int dims = {2,2};
dARRAY * A = randn(dims);

dARRAY * B = sum(A,0);
dARRAY * C = sum(A,1);
```

##### 1. 21. `mean()`

Finds and returns the mean of a matrix.

_Prototype_ :

```C
float * mean(dARRAY * matrix)
```

_Example_ :

```C
int dims = {2,2};
dARRAY * A = randn(dims);

printf("%f",mean(A));
```

##### 1. 22. `std()`

Finds and returns the standard deviation of a matrix.

_Prototype_ :

```C
float * std(dARRAY * matrix)
```

_Example_ :

```C
int dims = {2,2};
dARRAY * A = randn(dims);

printf("%f",std(A));
```

##### 1. 23. `var()`

Finds and returns the variance of a matrix.

_Prototype_ :

```C
float * var(dARRAY * matrix)
```

_Example_ :

```C
int dims = {2,2};
dARRAY * A = randn(dims);

printf("%f",var(A));
```

##### 1. 24. `permutation()`

Returns as array of integers with array values being shuffled natural numbers from 0 to length specified.

Useful for shuffling an array of elements.

_Prototype_ :

```C
int * permutation(int length)
```

_Example_ :

```C
int template = permutation(5);
```

_Output_ :
$\begin{bmatrix} 1 & 3 & 0 & 2 & 4\end{bmatrix}$

##### 1. 25. `frobenius_norm()`

Returns the frobenius norm of a matrix.

_Prototype_ :

```C
float frobenius_norm(dARRAY * matrix)
```

_Example_ :

```C
int dims = {5,5};
dARRAY * A = randn(dims);
float norm = forbenius_norm(A);
```

##### 1. 26. `Manhattan_distance()`

Returns the Manhattan distance of a matrix.

_Prototype_ :

```C
float Manhattan_distance(dARRAY * matrix)
```

_Example_ :

```C
int dims = {5,5};
dARRAY * A = randn(dims);
float dist = Manhattan_distance(A);
```

##### 1. 27. `free2d()`

Deallocates a matrix

_Prototype_ :

```C
void free2d(dARRAY * matrix);
```

_Example_ :

```C
int dims = {5,5};
dARRAY * A = randn(dims);
free2d(A);
A = NULL; //Just to be safe
```
