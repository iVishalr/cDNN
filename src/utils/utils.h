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
  dARRAY * power(dARRAY * matrix, int power);
  dARRAY * b_cast(dARRAY * MatrixA, dARRAY * MatrixB);
  dARRAY * reshape(dARRAY * matrix, int * dims);

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
#ifdef __cplusplus
}
#endif
#endif //UTILS_H