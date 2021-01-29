#include "../activations.h"

dARRAY * forward_pass_relu(dARRAY * layer_matrix){
  dARRAY * relu_out = (dARRAY*)malloc(sizeof(dARRAY));
  relu_out->matrix = (double*)calloc(layer_matrix->shape[0]*layer_matrix->shape[1],sizeof(double));
  for(int i=0;i<layer_matrix->shape[0]*layer_matrix->shape[1];i++)
    relu_out->matrix[i] = layer_matrix->matrix[i]>(double)0 ?(double)layer_matrix->matrix[i] : 0.0;
  relu_out->shape[0] = layer_matrix->shape[0];
  relu_out->shape[1] = layer_matrix->shape[1];
  return relu_out;
}

dARRAY * backward_pass_relu(dARRAY * layer_matrix){
  dARRAY * relu_out = (dARRAY*)malloc(sizeof(dARRAY));
  relu_out->matrix = (double*)calloc(layer_matrix->shape[0]*layer_matrix->shape[1],sizeof(double));
  for(int i=0;i<layer_matrix->shape[0]*layer_matrix->shape[1];i++)
    relu_out->matrix[i] = layer_matrix->matrix[i]>(double)0 ? 1.0 : 0.0;
  relu_out->shape[0] = layer_matrix->shape[0];
  relu_out->shape[1] = layer_matrix->shape[1];
  return relu_out;
}

ReLu * ReLu__init__(dARRAY * linear_matrix){
  ReLu * relu = (ReLu*)malloc(sizeof(ReLu));
  relu->forward = forward_pass_relu;
  relu->backward = backward_pass_relu;
  relu->in_dims[0] = relu->out_dims[0] = linear_matrix->shape[0];
  relu->in_dims[1] = relu->out_dims[1] = linear_matrix->shape[1];
  return relu;
}

dARRAY * (relu)(ReLu_args args){
  ReLu * r = ReLu__init__(args.input);
  if(!args.status)
    return r->forward(args.input);
  else
    return r->backward(args.input);
}

