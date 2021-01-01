#include "./relu.h"

dARRAY * forward_pass(dARRAY * layer_matrix){
  dARRAY * relu_out = (dARRAY*)malloc(sizeof(dARRAY));
  relu_out->matrix = (double*)malloc(sizeof(double)*layer_matrix->shape[0]*layer_matrix->shape[1]);
  for(int i=0;i<layer_matrix->shape[0]*layer_matrix->shape[1];i++)
    relu_out->matrix[i] = layer_matrix->matrix[i]>(double)0 ? layer_matrix->matrix[i] : 0;
  relu_out->shape[0] = layer_matrix->shape[0];
  relu_out->shape[1] = layer_matrix->shape[1];
  return relu_out;
}

dARRAY * backward_pass(dARRAY * layer_matrix){
  dARRAY * relu_out = (dARRAY*)malloc(sizeof(dARRAY));
  relu_out->matrix = (double*)malloc(sizeof(double)*layer_matrix->shape[0]*layer_matrix->shape[1]);
  for(int i=0;i<layer_matrix->shape[0]*layer_matrix->shape[1];i++)
    relu_out->matrix[i] = layer_matrix->matrix[i]>(double)0 ? 1 : 0;
  relu_out->shape[0] = layer_matrix->shape[0];
  relu_out->shape[1] = layer_matrix->shape[1];
  return relu_out;
}

ReLu * ReLu__init__(dARRAY * linear_matrix,int layer){
  ReLu * relu = (ReLu*)malloc(sizeof(ReLu));
  relu->forward_prop = forward_pass;
  relu->back_prop = backward_pass;
  relu->in_dims[0] = relu->out_dims[0] = linear_matrix->shape[0];
  relu->in_dims[1] = relu->out_dims[1] = linear_matrix->shape[1];
  relu->layer_num = layer;
  return relu;
}

