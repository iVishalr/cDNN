 /** 
  * File:    utils.h 
  * 
  * Author:  Vishal R (vishalr@pesu.pes.edu or vishalramesh01@gmail.com) 
  * 
  * Summary of File: 
  *   Matrix operation library required for cdnn. Contains all the header files for matrix operations. 
  */ 

#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <cblas.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <termcap.h>

#ifdef __cplusplus
extern "C"{
#endif

  typedef struct array{
    float * matrix;
    int shape[2];
  }dARRAY;

  dARRAY * zeros(int * dims);
  dARRAY * ones(int * dims);
  dARRAY * eye(int * dims);
  dARRAY * randn(int * dims);
  dARRAY * add(dARRAY * MatrixA, dARRAY * MatrixB);
  dARRAY * addScalar(dARRAY * matrix, float scalar);
  dARRAY * subtract(dARRAY * MatrixA, dARRAY * MatrixB);
  dARRAY * subScalar(dARRAY * matrix, float scalar);
  dARRAY * sum(dARRAY * matrix, int axis);
  dARRAY * transpose(dARRAY *  Matrix);
  dARRAY * dot(dARRAY *  MatrixA, dARRAY *  MatrixB);
  dARRAY * multiply(dARRAY * MatrixA, dARRAY * MatrixB);
  dARRAY * mulScalar(dARRAY * matrix, float scalar);
  dARRAY * divison(dARRAY * MatrixA, dARRAY * MatrixB);
  dARRAY * divScalar(dARRAY * matrix, float scalar);
  dARRAY * power(dARRAY * matrix, float power);
  dARRAY * squareroot(dARRAY * matrix);
  dARRAY * exponentional(dARRAY * matrix);
  dARRAY * b_cast(dARRAY * MatrixA, dARRAY * MatrixB);
  dARRAY * reshape(dARRAY * matrix, int * dims);

  int * permutation(int length);
  float mean(dARRAY * matrix);
  float var(dARRAY * matrix, char * type);
  float std(dARRAY * matrix, char * type);
  float gaussGenerator(float * cache, int * return_cache);
  float gaussRandom();
  float rand_norm(float mu, float std);

  float frobenius_norm(dARRAY * matrix);
  float Manhattan_distance(dARRAY * matrix);

  int size(dARRAY * A);
  void shape(dARRAY * A);
  void free2d(dARRAY * matrix);
  void sleep_my(int milliseconds);
  void cleanSTDIN();
  void get_safe_nn_threads();
#ifdef __cplusplus
}
#endif
#endif //UTILS_H