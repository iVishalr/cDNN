<p align="center">
<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/logo.png" alt="matrix" height="170px" width="550px"></img>
</p>

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

<p align="center">
<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/1.png" alt="matrix" height="90px" width="200px"></img>
</p>

The matrix has a shape `(3,3)`. The elements of the matrix would be stored in memory as follows :

<p align="center">
<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/2.png" alt="matrix" height="40px" width="350px"></img>
</p>

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

<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/3.png" alt="reshape_output" height="80px" width="550px"></img>

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

`9`

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

<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/4.png" alt="matrix" height="80px" width="150px"></img>

_Example_ :

```C
int A_dims = {3,3};
dARRAY * A = ones(A_dims);

int B_dims = {1,3};
dARRAY * B = ones(B_dims);

dARRAY * C = add(A,B);
```

_Output_ :

<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/5.png" alt="matrix" height="250px" width="500px"></img>

_Example_ :

```C
int A_dims = {4,4};
dARRAY * A = ones(A_dims);

int B_dims = {4,1};
dARRAY * B = ones(B_dims);

dARRAY * C = add(A,B);
```

_Output_ :

<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/6.png" alt="matrix" height="250px" width="500px"></img>

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

<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/7.png" alt="matrix" height="250px" width="500px"></img>

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

<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/8.png" alt="matrix" height="220px" width="600px"></img>

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

<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/9.png" alt="matrix" height="220px" width="600px"></img>

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

<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/10.png" alt="matrix" height="130px" width="350px"></img>

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

<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/11.png" alt="matrix" height="130px" width="350px"></img>

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

<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/12.png" alt="matrix" height="130px" width="350px"></img>

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

<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/13.png" alt="matrix" height="130px" width="350px"></img>

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

<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/14.png" alt="matrix" height="30px" width="150px"></img>

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

#### 2. Static Computation Graphs

cDNN uses static computation graphs for organizing and wireing up your neural networks. Using computation graphs makes gradient flow between layers of neural network easier to calculate.

The computation graph used here is a Directed Acyclic Graph (DAG) which is topologically sorted. Internally, cDNN uses a fancier version of doubly linked list where each node in the linked list can be of different type.

```C
enum layer_type {INPUT, DENSE, LOSS};
typedef struct computational_graph{
  struct computational_graph * next_layer;
  struct computational_graph * prev_layer;
  enum layer_type type;
  union
  {
    Dense_layer * DENSE;
    Input_layer * INPUT;
    loss_layer * LOSS;
  };
}Computation_Graph;
```

The `Computation_Graph` object contains has serveral functions that can be used to insert layers like `Dense` layer and so on into the linked list.

These functions will be used internally by the layers and the user need not worry about adding/deleting the layer to linked list.

#### 3. Neural Network

cDNN comes with a few layers like `Input(), Dense()` which can be used to create ANNs.

Each layer in cDNN follows a `Forward()` and `Backward()` API. The `Forward()` function is responsible for the behaviour of layer in the forward pass of the neural network. Similarly, `Backward()` is responsible for the behaviour of layer during the backward pass of the neural network. This is where gradient calculation and gradient flow takes place. Using this API it is easier to create scalable models as we need to just define the behaviour of a certain layer in the forward pass and backward pass rather than using one big equation to calculate loss and perform backpropagation.

General workflow in creating neural networks would be something like this. First you will take in the features and examples from your dataset as one big matrix. Traverse down the computation graph by passing the output of one layer as input of next layer and in the end compute a loss. Using the loss, we need to find the gradients of loss function with respect to all the parameters present in model. Using the API we discussed above, makes this process easier. This step is known as Backpropagation which will be discussed later. After backpropagtion, we need to use the computed gradients and perform parameter updates.

##### 3. 1. Forward Pass

```C
void __forward__(){
  Computation_Graph * temp = m->graph;
  while(temp!=NULL){
    m->current_layer = temp;
    if(temp->type==INPUT) temp->INPUT->forward();
    else if(temp->type==DENSE) temp->DENSE->forward();
    else if(temp->type==LOSS) temp->LOSS->forward();
    temp = temp->next_layer;
  }
}
```

In forward prop, we will traverse down the computation graph as shown above. This function calls `forward()` method of all the layers present in the model.

##### 3. 2. Backward Pass

```C
void __backward__(){
  Computation_Graph * temp = m->current_layer;
  while(temp->prev_layer!=NULL){
    m->current_layer = temp;
    if(temp->type==INPUT) temp->INPUT->backward();
    else if(temp->type==DENSE) temp->DENSE->backward();
    else if(temp->type==LOSS) temp->LOSS->backward();
    temp = temp->prev_layer;
  }
}
```

In backprop, we will traverse up the computation graph as shown above. This function calls `backward()` method of all the layers present in the model.

The backward pass is responsible for all the gradient calculations. The gradients calculated at the intermediate layers will flow through the layers until the very first layer using the chain rule of calculus.

Notice how we make use of the Forward/Backward API we discussed earlier. This is how famous libraries like PyTorch and Tensorflow implement neural networks internally.

We will look into how the gradients are calculated at the individual layers in the next section.

#### 4. Layers in cDNN

##### 4. 1. `Input()`

This layer is responsible for taking in the flattened datatset and feed it into the `Dense` layers.

_Example_ :

```C
Input(.layer_size=12288);
```

**Arguments** :

The `Input` layer takes in a required argument called `layer_size`.

1. `layer_size` - Denotes the number of nodes to be present in the `Input` Layer. `layer_size` must be equal to the number of features in your training set.

**Note** : The arguments to `Input()` must be preceded by a period (`.`). This is because `Input()` uses a structure to accept arguments from the user. The use of this method is made more clear in the upcoming layers.

**`Forward()`** :

```C
void forward_pass_input(){
  m->graph->INPUT->A = m->x_train_mini_batch[m->current_mini_batch];
}
```

**`Backward()`** :

```C
void backward_pass_input(){ }
```

We don't use backward pass for input layer hence we will leave it blank but keep the function just to satisfy our API.

_Internal Implementation_ :

```C
typedef struct input_layer{
  int input_features_size;
  dARRAY * A;
  __compute forward;
  __compute backward;
}Input_layer;

typedef struct input_args{
  int layer_size;
}Input_args;

void (Input)(Input_args input_layer_args){
  Input_layer * layer = (Input_layer*)malloc(sizeof(Input_layer));
  layer->input_features_size = input_layer_args.layer_size;
  layer->forward = forward_pass_input;
  layer->backward = backward_pass_input;
  //Append to computation graph
  append_graph(layer,"Input");
}
```

You can see how we are using a structure to pass arguments to the layer.

`__compute` is just a pointer to a function. This helps to create object oriented programming in C even though it is a procedural programming language. This is completely safe to do and authors or linux have also used this everywhere.

##### 4. 2. `Dense()`

`Dense()` layer performs a linear transformation

<p align="center">
<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/15.png" alt="matrix" height="30px" width="100px"></img>
</p>

on its inputs (`X`). It also applies dropout if it is enabled and pass the result to an activation function (`A`).

**Arguments** :

1. **`.layer_size`** - Specifies number of nodes in the layer.
   <br>
2. **`.layer_type`** - Specifies the type of `Dense` layer used.

   `.layer_type="hidden"` - the dense layer behaves as a hidden layer in the network.

   `.layer_type="output"` - the layer behaves as the last layer which is the output layer of the network.
   <br>

3. **`.activation`** - Specifies the type of activation function (A) to use for the layer.

   `.activation="relu"` - tells layer to use the 'ReLu' Activation function.

   <p align="center">
   <img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/16.png" alt="matrix" height="80px" width="210px"></img>
   </p>
   
   _Note_ : Please use `.initializer="he"` when using ReLu activation function.

   `.activation="sigmoid"` - tells layer to use the 'sigmoid' Activation function.

   <p align="center">
   <img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/17.png" alt="matrix" height="60px" width="200px"></img>
   </p>

   `.activation="tanh"` - tells layer to use the 'tanh' Activation function.

   <p align="center">
   <img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/18.png" alt="matrix" height="60px" width="200px"></img>
   </p>
   
   _Note_ : Please use `.initializer="xavier"` when using tanh activation function.

   `.activation="softmax"` - tells layer to apply a 'softmax' Activation function.

   <p align="center">
   <img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/19.png" alt="matrix" height="70px" width="200px"></img>
   </p>
   
   _Note_ : Please use `.initializer="random"` when using Softmax activation function.
   
   <br>

4. **`.dropout`** - Specifies the dropout to be used in the layer.

   `.dropout=1.0` - No dropout will be applied.

   `.dropout=0.5` - Specifies that there is a 50% chance of dropping out certain nodes in the layer.

   `.dropout=x` - Specifies that there is a x\*100% chance of dropping out certain nodes in the layer.

   `x` must be within 0.0 and 1.0

   By default,`Dense()` will use `.dropout=1.0`.
   <br>

5. **`.initializer`** - Specifies the type of initialization to be used for initializing the layer weights.

   `.initializer="he"` - 'He' initialization will be used for weight initialization.

   <p align="center">
   <img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/20.png" alt="matrix" height="80px" width="180px"></img>
   </p>

   `.initializer="xavier"` - 'Xavier' initialization will be used for weight initialization.

   <p align="center">
   <img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/21.png" alt="matrix" height="80px" width="180px"></img>
   </p>

   `.initializer="random"` - Weights will be intialized to random values using normal distribution.

   <p align="center">
   <img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/22.png" alt="matrix" height="30px" width="150px"></img>
   </p>

   `.initializer="zeros"` - Weights will be set to zero. **Don't use this option**. It is just there to show that network fails to break symmetry when `W=0`.

   By default,`Dense()` will use `.initializer="he"`.

_Example_:

```C
Dense(.layer_size=32,.activation="relu",.layer_type="hidden",.dropout=0.5);
```

In the above example, we are defining a Dense layer with 32 neurons/nodes. We apply `relu` activation and a dropout of 0.5.

**`Forward()`** :

For the above example, the forward pass would be something like this :

<p align="center">
<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/24.png" alt="matrix" height="110px" width="250px"></img>
</p>

**`Backward()`** :

For the above example, the backward pass would be something like this :

<p align="center">
  <img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/25.png" alt="matrix" height="120px" width="300px"></img>
</p>

_Note_ : the above backprop equations are just for illustrations. There may be dimension mismatches.

The basic idea employed here is, if we are traversing the computation graph from bottom to top, the gradient flow would be somthing like this :
For a function `f(x)`,

1. We will find the `local gradients` :

   <p align="center">
   <img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/26.png" alt="matrix" height="70px" width="400px"></img>
   </p>

2. We will have a gradient from below called `global gradients`.
3. Using chain rule of calculus,

   <p align="center">
   <img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/27.png" alt="matrix" height="230px" width="400px"></img>
   </p>

Now dx, dy, dz are gradients that are 'flowing' into inputs x, y, and z. Gradients dx, dy, dz become a global gradient for functions present above f(x). Remember, we are traversing the computation graph from last node to first node. That's why we are refering to gradients coming from below and flowing to above functions.

It's kind of difficult to explain this without a computation graph.

##### 4. 3. `Model()`

This layer is responsible for putting the model together. Basically it combines all the layers and initializes the weights according to the initializer used. It also adds a loss function to the computation graph and initializes the optimizer's internal state.

**Arguments** :

1. **`.X_train`** - Specifies a pointer to the training set containing features.
2. **`.y_train`** - Specifies a pointer to the training set containing true labels.
3. **`.X_cv`** - Specifies a pointer to the validation set containing features.
4. **`.y_cv`** - Specifies a pointer to the validation set containing true labels.
5. **`.X_test`** - Specifies a pointer to the test set containing features.
6. **`.y_test`** - Specifies a pointer to the test set containing true labels.
7. **`.epochs`** - Specifies the number of epochs the model must perform.
8. **`.batch_size`** - Specifies the batch_size for the model.
9. **`.optimizer`** - Specifies the optimizer to be used for training.

   `.optimizer="adam"` - Uses the Adam optimization for parameter updates.

   `.optimizer="adagrad"` - Uses the Adagrad optimization for parameter updates.

   `.optimizer="rmsprop"` - Uses the RMSProp optimization for parameter updates.

   `.optimizer="momentum"` - Uses the Momentum optimization for parameter updates.

   `.optimizer="sgd"` - Uses the Gradient Descent optimization for parameter updates.

   By default, `.optimizer="adam"`.

10. **`.regularization`** - Specifies the type of regularization to be used for training.

    `.regularization="L1"` - Uses the L1 regularization for training.

    `.regularization="L2"` - Uses the L2 regularization for training.

    By default, `.regularization="L2"`.

11. **`.weight_decay`** - Specifies the regularization strength for training.
12. **`.lr`** - Specifies the learning rate for training.
13. **`.beta`** - Specifies the decay value that will be used during parameter updates.

    By default, `.beta=0.9`.

14. **`.beta1`** - Specifies the decay value that will be used for first order moment calculations during parameter updates.

    By default, `.beta1=0.9`.

15. **`.beta2`** - Specifies the decay value that will be used for second order moment calculations during parameter updates.

    By default, `.beta2=0.999`.

16. **`.loss`** - Specifies the loss function to be used for training.

    `.loss="cross_entropy_loss"` - Uses the CrossEntropyLoss function for training.

    `.loss="MSELoss"` - Uses the MSELoss function for training.

    By default, `.loss="cross_entropy_loss"`.

17. **`.checkpoint_every`** - Specifies how often (in epochs) the model must be saved.

    By default, `.checkpoint_every=2500`. (2500th epoch).

All the above arguments except `.X_train,y_train` are optional arguments. To achieve this functionality, we use structures to accept the arguments from the user and pass it to the model. The user cannot remember all the arguments that must be provided hence this method is the only suitable way.

Default initalization for all arguments is given below :

```C
Model(...) Model((Model_args){\
.X_train=NULL,.y_train=NULL,\
.X_cv=NULL,.y_cv=NULL,\
.X_test=NULL,.y_test=NULL,\
.epochs=10,\
.batch_size=64,\
.optimizer="Adam",\
.regularization=NULL,\
.weight_decay=0.0,\
.lr=3e-4,\
.print_cost=1,\
.beta=0.9,\
.beta1=0.9,\
.beta2=0.999,\
.loss="cross_entropy_loss",\
.checkpoint_every=2500,__VA_ARGS__});
```

#### 5. Model Saving/Loading

##### 5. 1. Saving the Model

After the model has been trained, the model paramters can be saved by using the following function.

```C
Save_Model(char * filename)
```

_Example_ :

```C
Save_Model("./model/DOGS_VS_CATS.t7");
```

Please use `.t7` format for loading and saving model.

##### 5. 2. Loading the Model

A trained model can be loaded in by using the following function.

```C
Load_Model(char * filename)
```

_Example_ :

```C
Load_Model("./model/DOGS_VS_CATS.t7");
```

Please use `.t7` format for loading and saving model.

#### 6. Loading Dataset into Memory

The following functions can be used to load the dataset into memory for training.

```C
dARRAY * load_x_train(char * filename, int * dims);
dARRAY * load_y_train(char * filename, int * dims);
dARRAY * load_x_cv(char * filename, int * dims);
dARRAY * load_y_cv(char * filename, int * dims);
dARRAY * load_x_test(char * filename, int * dims);
dARRAY * load_y_test(char * filename, int * dims);
```

_Example_ :

```C
#include <cdnn.h>

int main(){

  Create_Model(); //used to create a Model object. Don't miss to include this.

  int x_train_dims[] = {12288,100};
  int y_train_dims[] = {2,100};
  int x_cv_dims[] = {12288,100};
  int y_cv_dims[] = {2,100};
  int x_test_dims[] = {12288,100};
  int y_test_dims[] = {2,100};

  dARRAY * x_train = load_x_train("./data/X_train.t7",x_train_dims);
  dARRAY * y_train = load_y_train("./data/y_train.t7",y_train_dims);
  dARRAY * x_cv = load_x_cv("./data/X_cv.t7",x_cv_dims);
  dARRAY * y_cv = load_y_cv("./data/y_cv.t7",y_cv_dims);
  dARRAY * x_test = load_x_test("./data/X_test.t7",x_test_dims);
  dARRAY * y_test = load_y_test("./data/y_test.t7",y_test_dims);

  return 0;
}
```

The loaded dataset can be passed to the model as shown below. Here we are creating a 2-layer neural network.

```C
#include <cdnn.h>

int main(){

  Create_Model();

  int x_train_dims[] = {12288,100};
  int y_train_dims[] = {2,100};
  int x_cv_dims[] = {12288,100};
  int y_cv_dims[] = {2,100};
  int x_test_dims[] = {12288,100};
  int y_test_dims[] = {2,100};

  dARRAY * x_train = load_x_train("./data/X_train.t7",x_train_dims);
  dARRAY * y_train = load_y_train("./data/y_train.t7",y_train_dims);
  dARRAY * x_cv = load_x_cv("./data/X_cv.t7",x_cv_dims);
  dARRAY * y_cv = load_y_cv("./data/y_cv.t7",y_cv_dims);
  dARRAY * x_test = load_x_test("./data/X_test.t7",x_test_dims);
  dARRAY * y_test = load_y_test("./data/y_test.t7",y_test_dims);

  Input(.layer_size=12288);
  Dense(.layer_size=64,.activation="relu",.initializer="he",.layer_type="hidden");
  Dense(.layer_size=2,.activation="softmax",.initializer="random",.layer_type="output");
  Model(.X_train=x_train,.y_train=y_train,.X_cv=x_cv,.y_cv=y_cv,.X_test=x_test,.y_test=y_test,.epochs=1000,.lr=3.67e-5,.optimizer="adam");

  return 0;
}
```

#### 7. Training and Testing the Model

##### 7. 1. Training the Model

`Fit()` can be used to train the model.

_Example_ :

```C
#include <cdnn.h>

int main(){
  ...
  //include the statements for creating model obj and loading data set here.
  ...
  Input(.layer_size=12288);
  Dense(.layer_size=64,.activation="relu",.initializer="he",.layer_type="hidden");
  Dense(.layer_size=2,.activation="softmax",.initializer="random",.layer_type="output");
  Model(.X_train=x_train,.y_train=y_train,.X_cv=x_cv,.y_cv=y_cv,.X_test=x_test,.y_test=y_test,.epochs=1000,.lr=3.67e-5,.optimizer="adam");

  Fit();

  return 0;
}
```

##### 7. 2. Testing the Model

`Test()` can be used to test the model on your test_set.

_Example_ :

```C
...

Input(.layer_size=12288);
Dense(.layer_size=64,.activation="relu",.initializer="he",.layer_type="hidden");
Dense(.layer_size=2,.activation="softmax",.initializer="random",.layer_type="output");
Model(.X_train=x_train,.y_train=y_train,.X_cv=x_cv,.y_cv=y_cv,.X_test=x_test,.y_test=y_test,.epochs=1000,.lr=3.67e-5,.optimizer="adam");

Fit();
Test();
```

##### 7. 3. Testing on individual images

`load_image()` can be used to load an image into memory.
`Predict()` can be used for getting the model predictions.

_Prototype_ :

```C
  dARRAY * Predict(dARRAY * image, int verbose)
```

verbose - If verbose=1, cDNN prints the output of model.

Return value of `Predict()` is a pointer to the output of the model.

_Example_ :

```C
...

Input(.layer_size=12288);
Dense(.layer_size=64,.activation="relu",.initializer="he",.layer_type="hidden");
Dense(.layer_size=2,.activation="softmax",.initializer="random",.layer_type="output");
Model(.X_train=x_train,.y_train=y_train,.X_cv=x_cv,.y_cv=y_cv,.X_test=x_test,.y_test=y_test,.epochs=1000,.lr=3.67e-5,.optimizer="adam");

Fit();
Test();

dARRAY * test_img1 = load_image("test_img1.data");
dARRAY * test_img2 = load_image("test_img2.data");

dARRAY * img1_score = Predict(test_img1,1);
dARRAY * img2_score = Predict(test_img2,1); //1 is for verbose
```

#### 8. Plotting Model Metrics

cDNN does not provide any plotting tools. However, the model metrics such as training loss, training accuracy and validation accuracy will be dumped to a file on disk by default. These can be read into memory in Python and using Matplotlib, we can visualize the curves.

#### 9. Early Stopping

There is an option to run your model infinitely. Setting `.epochs=-1` runs the model forever. To stop training, press `CTRL + C`. cDNN will ask you if you want to save the model or not. Please follow the menu options to save the model and to give it a name.

#### 10. Additional Notes

1. `Adam(), RMSProp() and Adagrad()` are powerful optimizers. If you have used a high learning rate, there is a possibility of getting `nan or inf` as the cost after a while during training. If you get it during training, just stop the training process as there's no point in continuing further. Reinitialize your `lr` to a smaller value and try again.

Setting learning rate correctly is important. For me, `lr > 4.65e-5` was working well.

2. Remember to set the `.layer_size` to be equal to the number of features you are using in training set.

3. `Softmax()` outputs a tensor with `num_of_classes` elements. If you are trying to train say CIFAR-10, your output layer must have 10 as `.layer_size` as there are 10 classes in the dataset. Otherwise you may get something else as the output.

4. While organizing your training, val and test sets, remember that cDNN puts your features along rows. Meaning, it stacks your examples along columns and the rows will represent features.

<p align="center">
<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/28.png" alt="matrix" height="130px" width="350px"></img>
</p>

Here, there are `n` examples stacked column wise and each example has `m` features which are stacked row wise.

5. There are no pre-processing functions available in cDNN. Please do it in Python and extract the pixel values one by one if you are working with images and write it to a `.t7` file and then load it in using cDNN functions.
6. Visualizing model metrics can be done in Python by reading the data present in the `./bin/` folder.
