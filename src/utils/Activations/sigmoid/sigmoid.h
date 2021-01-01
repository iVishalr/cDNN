#include "../../utils.h"

typedef dARRAY * (*forward)(dARRAY * linear_matrix);
typedef dARRAY * (*backward)(dARRAY * linear_matrix);

typedef struct sigmoid{
  int in_dims[2];
  int out_dims[2];
  int layer_num;
  forward forward_prop;
  backward back_prop;
}Sigmoid;

Sigmoid * Sigmoid__init__(dARRAY * layer_matrix, int layer);