#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "../utils.h"

typedef struct{
  dARRAY * input;
  int status;
}ReLu_args;

typedef struct{
  dARRAY * input;
  int status;
}Sigmoid_args;

typedef struct{
  dARRAY * input;
  int status;
}Tanh_args;

#ifdef __cplusplus
extern "C"{
#endif
  typedef dARRAY* (*__forward)(dARRAY * layer_matrix);
  typedef dARRAY* (*__backward)(dARRAY * layer_matrix);

  typedef struct relu{
    int in_dims[2];
    int out_dims[2];
    __forward forward;
    __backward backward;
  }ReLu;

  typedef struct sigmoid{
    int in_dims[2];
    int out_dims[2];
    __forward forward;
    __backward backward;
  }Sigmoid;

  typedef struct tanh{
    int in_dims[2];
    int out_dims[2];
    __forward forward;
    __backward backward;
  }Tanh;

  ReLu * ReLu__init__(dARRAY * linear_matrix);
  Sigmoid * Sigmoid__init__(dARRAY * layer_matrix);
  Tanh * Tanh__init__(dARRAY * layer_matrix);

  dARRAY * (relu)(ReLu_args args);
  dARRAY * (sigmoid)(Sigmoid_args args);
  dARRAY * (TanH)(Tanh_args args);

#ifdef __cplusplus
}
#endif

#define relu(...) relu((ReLu_args){.input=NULL,.status=0,__VA_ARGS__});
#define sigmoid(...) sigmoid((Sigmoid_args){.input=NULL,.status=0,__VA_ARGS__});
#define TanH(...) TanH((Tanh_args){.input=NULL,.status=0,__VA_ARGS__});

#endif //ACTIVATIONS_H