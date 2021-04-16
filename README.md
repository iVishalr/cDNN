<p align="center">
<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/logo.png" alt="matrix" height="170px" width="550px"></img>
</p>

**cDNN** is a Deep Learning Library written in C Programming Language. cDNN provides functions that can be used to create Artificial Neural Networks (ANN). These functions are designed to be as efficient as possible both in performance and memory.

## Features

1. **cDNN** provides simple functions for creating ANNs.
2. These functions are designed to achieve maximum performance possible on a cpu.
3. At the core, the matrix library provides basic matrix-matrix operations which are required to implement neural networks.
4. These matrix-matrix operations are very efficient and is as fast as some of the popular scientific computing libraries like Numpy.
5. cDNN uses a Static Computation Graphs (DAGs) to wireup your ANNs and to perform gradient calculations.
6. The library also provides helper functions that can be used by the user to save models, print graphs and so on.

## Documentation

The documentation of cDNN is available [here](https://github.com/iVishalr/cDNN/blob/main/documentation/DOCUMENTATION.md). I have tried to document it as extensively as possible. Feel free to modify or correct it.

## More about cDNN

### 1. Matrix Library

This is the heart of the entire library. It provides basic matrix-matrix operations required for the basic functioning of neural networks. These functions are designed to be as efficient as possible without performance tradeoffs.

Before we go deeper on how these matrix-matrix operations occur, we need to take a look at how a matrix is created and organized in memory.

```C
typedef struct array{
  float * matrix;
  int shape[2];
}dARRAY;
```

The above structure data type is used to create matrices. `float * matrix` stores the elements of the matrix and `int shape[2]` stores the shape or the order of the matrix.

The elements of a matrix are stored in a RowMajor fashion in memory.

Consider the following matrix :

<p align="center">
<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/1.png" alt="matrix" height="100px" width="200px" ></img>
</p>

The matrix has a shape `(3,3)`. The elements of the matrix would be stored in memory as follows :

<p align="center">
<img src="https://github.com/iVishalr/cDNN/blob/main/documentation/images/2.png" alt="matrix" height="40px" width="350px" ></img>
</p>

`float * matrix` stores the above array and `int shape[2] = {3,3}`. The shape of the matrix helps us to know the dimensions of the matrix and helps to perform matrix-matrix operations accordingly.

The main advantage of this type of matrix organization is that it eliminates the use of `float ** matrix` to store a 2D matrix. Operations would be very slow if we used `float ** matrix` due to double lookup table in memory.

Access of elements of a matrix in RowMajor order can be done using the following way

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

Now comming to the main topic, the matrix-matrix operations are performed in two ways :

1. Using efficient `BLAS` operations.
2. Using parallelized loop operations.

Additional details are available in documentation.

The things discussed above help us to create neural networks and perform gradient calculations.

### 2. Static Computation Graphs

cDNN uses a static computation graph to wireup your neural networks. Popular deep learning libraries like PyTorch, Tensorflow, Caffe ... go even further and use dynamic computation graphs. Dynamic graphs are difficult to implement hence, we will only use static graphs in this library.

Fun fact, Tensorflow 1.0 used Static Computation Graphs. Tensorflow 2.0 introduced Dynamic Computation Graphs.

### 3. Performance

cDNN is as fast as Numpy or even faster than Numpy in some cases. This makes model training so much quicker and helps you iterate over models very quickly.

Major performance boost comes from implementing certain matrx-matrix functions like matrix multiplication in fortran. cDNN replies upon `BLAS` provided by `OpenBLAS` to perform certain operations in a highly efficient way.

cDNN also uses automatic thread calculations and executes matrix operations in parallel that don't use BLAS to achieve parallelization, cDNN relies on OpenMP to aid in thread creation process and other thread issues like synchronization.

## Installation

Requirements,

1. gcc
2. ncurses
3. Openblas
4. OpenMP

### Installing the Dependencies

On Linux,

```bash
$ sudo apt-get install gcc
$ sudo apt-get install gfortran  #important don't miss this!
$ sudo apt-get install libomp-dev
$ sudo apt-get install libncurses-dev

# Downloading OpenBlas from Source. There's no other way to install OpenBlas in how cDNN wants it.

$ git clone https://github.com/xianyi/OpenBLAS.git
$ cd OpenBlas
$ sudo make && sudo make install #This will take a while depending on your system speed. You may see some warnings along the way. Don't worry about it.
```


On macOS,

```bash
$ brew install ncurses
$ brew install gfortran
$ git clone https://github.com/xianyi/OpenBLAS.git
$ cd OpenBlas
$ sudo make && sudo make install
```

Installing OpenBLAS from their source will take a while. If you run into any errors while installing OpenBLAS, please refer to their [User Manual](https://github.com/xianyi/OpenBLAS/wiki/User-Manual).

### Building cDNN

After installing the dependecies, execute the following in terminal.

```bash
$ git clone https://github.com/iVishalr/cDNN.git
$ cd cDNN
$ sudo make && sudo make install
```

This will create a shared library in your system which allows you to use the fuctions in cDNN anywhere. You do not need to have the source code with you after the shared library has been created and stored in system.

`$ sudo make && sudo make install` will create a shared library according to the platform you are using and will place the library in `/usr/local/lib` and the include header files will be placed in `/usr/local/include`.

Note : Please do not change anything in `Makefile` of cDNN as you will be installing in the standard directories where other shared libraries like `libc.so` and so on will be present. You may risk modifying/deleting other libraries in your system if you change things in `Makefile`.

I know its a lot of work, but there's no way around it.

## Compiling

To compile a `test.c` file that uses cDNN, please type the following in terminal

On Linux,
```bash
$ gcc test.c -lcdnn -lncurses -lopenblas -lgomp -I /usr/local/include -L /usr/local/lib -I /opt/OpenBLAS/include/ -L /opt/OpenBLAS/lib/ -lm
```
Please keep the above LDFLAGS (`-lcdnn`,`-lopenblas`, ....) in the same order. Otherwise test.c won't compile.

On macOS,
```bash
$ gcc test.c -lcdnn -I /usr/local/include/ -L /usr/local/lib/ -I /opt/OpenBLAS/include/ -L /opt/OpenBLAS/lib/
```

Since the shared library depends on OpenBLAS's implementation of `cblas.h`, you are requried to include its header files as well as its shared library.

To run the program, execute `./a.out` or `./<name of your executable>`

## Examples

```C
/*
File : test.c
Author : Vishal R
Email ID : vishalr@pesu.pes.edu or vishalramesh01@gmail.com
Abstract : Implements a 5 layer neural network using cDNN
*/

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
  Dense(.layer_size=32,.activation="relu",.initializer="he",.layer_type="hidden",.dropout=0.7);
  Dense(.layer_size=32,.activation="relu",.layer_type="hidden",.dropout=0.5);
  Dense(.layer_size=16,.activation="relu",.layer_type="hidden");
  Dense(.layer_size=2,.activation="softmax",.initializer="random",.layer_type="output");
  Model(.X_train=x_train,.y_train=y_train,.X_cv=x_cv,.y_cv=y_cv,.X_test=x_test,.y_test=y_test,\
        .epochs=1000,.lr=3.67e-5,.optimizer="adam",.checkpoint_every=-1,.batch_size=32);

  Fit();
  Test();
  Save_Model("./DOGS_VS_CATS.t7");
  
  int img_dims[] = {12288,1};
  dARRAY * test_img1 = load_image("./test_img1.data",img_dims);
  dARRAY * test_img2 = load_image("./test_img2.data",img_dims);

  dARRAY * prediction1 = Predict(test_img1,1);
  dARRAY * prediction2 = Predict(test_img2,1);
  
  free2d(test_img1);
  free2d(test_img2);
  free2d(prediction1);
  free2d(prediction2);

  Destroy_Model();
}
```

Above file shows how to create a 5 layer neural network using cDNN library.

Additional examples are available in the [Examples](https://github.com/iVishalr/cDNN/blob/main/examples) folder.

## Contributions

If you like this library and would like to make it better, you are free to do so. It takes a team effort to make things better. Hence I would love to have you on board.

Avoid making commits directly to `main branch`. Create your own branch and make a pull request. After your pull request is approved, the changes you have made would be merged with the main code.

## License

cDNN has a MIT-style license, as found in [LICENSE](https://github.com/iVishalr/cDNN/blob/main/LICENCE) file.

