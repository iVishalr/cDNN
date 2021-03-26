#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "utils.h"

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

typedef struct{
  dARRAY * input;
  int status;
}softmax_args;

#ifdef __cplusplus
extern "C"{
#endif
  typedef dARRAY * (*__compute_act)();

  typedef struct relu{
    int in_dims[2];
    int out_dims[2];
    __compute_act forward;
    __compute_act backward;
  }ReLu;

  typedef struct sigmoid{
    int in_dims[2];
    int out_dims[2];
    __compute_act forward;
    __compute_act backward;
  }Sigmoid;

  typedef struct tanh{
    int in_dims[2];
    int out_dims[2];
    __compute_act forward;
    __compute_act backward;
  }Tanh;

  typedef struct softmax{
    int in_dims[2];
    int out_dims[2];
    __compute_act forward;
    __compute_act backward;
  }Softmax;

  ReLu * ReLu__init__(dARRAY * linear_matrix);
  Sigmoid * Sigmoid__init__(dARRAY * layer_matrix);
  Tanh * Tanh__init__(dARRAY * layer_matrix);
  Softmax * Softmax__init__(dARRAY * layer_matrix);

  dARRAY * (relu)(ReLu_args args);
  dARRAY * (sigmoid)(Sigmoid_args args);
  dARRAY * (TanH)(Tanh_args args);
  dARRAY * (softmax)(softmax_args args);

  dARRAY * forward_pass_relu();
  dARRAY * forward_pass_sigmoid();
  dARRAY * forward_pass_tanh();
  dARRAY * forward_pass_softmax();
  dARRAY * backward_pass_relu();
  dARRAY * backward_pass_sigmoid();
  dARRAY * backward_pass_tanh();
  dARRAY * backward_pass_softmax();

#ifdef __cplusplus
}
#endif

#define relu(...) relu((ReLu_args){.input=NULL,.status=0,__VA_ARGS__});
#define sigmoid(...) sigmoid((Sigmoid_args){.input=NULL,.status=0,__VA_ARGS__});
#define TanH(...) TanH((Tanh_args){.input=NULL,.status=0,__VA_ARGS__});
#define softmax(...) softmax((softmax_args){.input=NULL,.status=0,__VA_ARGS__});

#endif //ACTIVATIONS_H