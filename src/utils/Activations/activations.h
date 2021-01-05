#include "../utils.h"

typedef dARRAY* (*forward)(dARRAY * layer_matrix);
typedef dARRAY* (*backward)(dARRAY * layer_matrix);

typedef struct relu{
  int in_dims[2];
  int out_dims[2];
  forward forward_prop;
  backward back_prop;
}ReLu;

typedef struct sigmoid{
  int in_dims[2];
  int out_dims[2];
  forward forward_prop;
  backward back_prop;
}Sigmoid;

typedef struct tanh{
  int in_dims[2];
  int out_dims[2];
  forward forward_prop;
  backward back_prop;
}Tanh;

ReLu * ReLu__init__(dARRAY * linear_matrix);
Sigmoid * Sigmoid__init__(dARRAY * layer_matrix);
Tanh * Tanh__init__(dARRAY * layer_matrix);

typedef struct{
  dARRAY * input;
}ReLu_args;

typedef struct{
  dARRAY * input;
}Sigmoid_args;

typedef struct{
  dARRAY * input;
}Tanh_args;

#define relu(...) relu((ReLu_args){.input=NULL,__VA_ARGS__});
#define sigmoid(...) sigmoid((Sigmoid_args){.input=NULL,__VA_ARGS__});
#define TanH(...) TanH((Tanh_args){.input=NULL,__VA_ARGS__});

dARRAY * (relu)(ReLu_args args);
dARRAY * (sigmoid)(Sigmoid_args args);
dARRAY * (TanH)(Tanh_args args);

void display(dARRAY * linear_matrix);
