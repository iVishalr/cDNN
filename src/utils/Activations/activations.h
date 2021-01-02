#include "../utils.h"

typedef dARRAY* (*forward)(dARRAY * layer_matrix);
typedef dARRAY* (*backward)(dARRAY * layer_matrix);

typedef struct relu{
  int in_dims[2];
  int out_dims[2];
  int layer_num;
  forward forward_prop;
  backward back_prop;
}ReLu;

typedef struct sigmoid{
  int in_dims[2];
  int out_dims[2];
  int layer_num;
  forward forward_prop;
  backward back_prop;
}Sigmoid;

typedef struct tanh{
  int in_dims[2];
  int out_dims[2];
  int layer_num;
  forward forward_prop;
  backward back_prop;
}Tanh;

ReLu * ReLu__init__(dARRAY * linear_matrix,int layer);
Sigmoid * Sigmoid__init__(dARRAY * layer_matrix, int layer);
Tanh * Tanh__init__(dARRAY * layer_matrix, int layer);

void display(dARRAY * linear_matrix);
