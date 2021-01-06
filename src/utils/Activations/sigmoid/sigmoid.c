#include "../activations.h"

dARRAY * forward_pass_sigmoid(dARRAY * linear_matrix){
  dARRAY * sigmoid_out = (dARRAY*)malloc(sizeof(dARRAY));
  sigmoid_out->matrix = (double*)malloc(sizeof(double)*linear_matrix->shape[0]*linear_matrix->shape[1]);
  omp_set_num_threads(4);
  #pragma omp parallel for shared(linear_matrix,sigmoid_out)
  for(int i=0;i<linear_matrix->shape[0]*linear_matrix->shape[1];i++)
    sigmoid_out->matrix[i] = 1/(1+exp(-1*linear_matrix->matrix[i]));
  sigmoid_out->shape[0] = linear_matrix->shape[0];
  sigmoid_out->shape[1] = linear_matrix->shape[1];
  return sigmoid_out;
}

dARRAY * backward_pass_sigmoid(dARRAY * linear_matrix){
  dARRAY * sigmoid_out;
  omp_set_num_threads(4);
  int dims[] = {linear_matrix->shape[0],linear_matrix->shape[1]};
  sigmoid_out = multiply(linear_matrix,subtract(ones(dims),linear_matrix));
  sigmoid_out->shape[0] = linear_matrix->shape[0];
  sigmoid_out->shape[1] = linear_matrix->shape[1];
  return sigmoid_out;
}

Sigmoid * Sigmoid__init__(dARRAY * layer_matrix){
  Sigmoid * sigmoid = (Sigmoid*)malloc(sizeof(Sigmoid));
  sigmoid->forward_prop = forward_pass_sigmoid;
  sigmoid->back_prop = backward_pass_sigmoid;
  sigmoid->in_dims[0] = sigmoid->out_dims[0] = layer_matrix->shape[0];
  sigmoid->in_dims[1] = sigmoid->out_dims[1] = layer_matrix->shape[1];
  return sigmoid;
}


dARRAY * (sigmoid)(Sigmoid_args args){
  Sigmoid * s = Sigmoid__init__(args.input);
  if(!args.status)
    return s->forward_prop(args.input);
  else
    return s->back_prop(args.input);
}