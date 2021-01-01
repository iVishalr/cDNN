#include "../../utils.h"

typedef dARRAY* (*forward)(dARRAY * layer_matrix);
typedef dARRAY* (*backward)(dARRAY * layer_matrix);

typedef struct relu{
  int in_dims[2];
  int out_dims[2];
  int layer_num;
  forward forward_prop;
  backward back_prop;
}ReLu;

ReLu * ReLu__init__(dARRAY * linear_matrix,int layer);