#include "../activations.h"

dARRAY * forward_pass_tanh(dARRAY * layer_matrix){
  dARRAY * tanh_out = (dARRAY*)malloc(sizeof(dARRAY));
  tanh_out->matrix = (double*)malloc(sizeof(double)*layer_matrix->shape[0]*layer_matrix->shape[1]);
  // omp_set_num_threads(4);
  #pragma omp parallel for shared(layer_matrix,tanh_out)
  for(int i=0;i<layer_matrix->shape[0]*layer_matrix->shape[1];i++){
    //Computing the tanh function
    double exp_res1 = exp(layer_matrix->matrix[i]);
    double exp_res2 = exp(-1*layer_matrix->matrix[i]);
    tanh_out->matrix[i] = (exp_res1 - exp_res2)/(exp_res1 + exp_res2);
  }
  tanh_out->shape[0] = layer_matrix->shape[0];
  tanh_out->shape[1] = layer_matrix->shape[1];
  return tanh_out;
}

dARRAY * backward_pass_tanh(dARRAY * layer_matrix){
  dARRAY * tanh_out = (dARRAY*)malloc(sizeof(dARRAY));
  tanh_out->matrix = (double*)malloc(sizeof(double)*layer_matrix->shape[0]*layer_matrix->shape[1]);
  // omp_set_num_threads(4);
  #pragma omp parallel for shared(layer_matrix,tanh_out)
  for(int i=0;i<layer_matrix->shape[0]*layer_matrix->shape[1];i++){
    //Computing the tanh function
    double exp_res1 = exp(layer_matrix->matrix[i]);
    double exp_res2 = exp(-1*layer_matrix->matrix[i]);
    double temp = (exp_res1 - exp_res2)/(exp_res1 + exp_res2);
    //gradient of tanh is g'(z) = 1 - (tanh(z))^2
    tanh_out->matrix[i] = 1-pow(temp,2);
  }
  tanh_out->shape[0] = layer_matrix->shape[0];
  tanh_out->shape[1] = layer_matrix->shape[1];
  return tanh_out;
}

Tanh * Tanh__init__(dARRAY * linear_matrix){
  Tanh * tanh = (Tanh*)malloc(sizeof(Tanh));
  tanh->forward_prop = forward_pass_tanh;
  tanh->back_prop = backward_pass_tanh;
  tanh->in_dims[0] = tanh->out_dims[0] = linear_matrix->shape[0];
  tanh->in_dims[1] = tanh->out_dims[1] = linear_matrix->shape[1];
  return tanh;
}

dARRAY * (TanH)(Tanh_args args){
  Tanh * t = Tanh__init__(args.input);
  if(!args.status)
    return t->forward_prop(args.input);
  else 
    return t->back_prop(args.input);
}