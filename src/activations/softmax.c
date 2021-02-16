#include "./activations.h"
#include "../model/model.h"

extern __Model__ * m;

dARRAY * forward_pass_softmax(){
  dARRAY * softmax_outf = NULL;
  // printf("exping\n");
  dARRAY * exp_sub_max = exponentional(m->current_layer->DENSE->cache);

  dARRAY * div_factor = (dARRAY*)malloc(sizeof(dARRAY));
  div_factor->matrix = (float*)calloc(exp_sub_max->shape[1],sizeof(float));
  
  dARRAY * temp = transpose(exp_sub_max);
  // printf("calculating div factor\n");
  omp_set_num_threads(8);
  #pragma omp parallel for num_threads(8) collapse(1) shared(temp,div_factor) schedule(static)
  for(int i=0;i<temp->shape[0];i++){
    float sum_of_exps=0.0;
    for(int j=0;j<temp->shape[1];j++){
      sum_of_exps+= temp->matrix[i*temp->shape[1]+j];
    }
    div_factor->matrix[i] = sum_of_exps;
  }
  div_factor->shape[0] = 1;
  div_factor->shape[1] = exp_sub_max->shape[1];
  // printf("calculated div factor\n");
  softmax_outf = divison(exp_sub_max,div_factor);
  // printf("calculated sfmax output\n");
  
  free2d(temp);
  free2d(exp_sub_max);
  free2d(div_factor);
  temp = exp_sub_max = div_factor = NULL; 
  return softmax_outf;
}

dARRAY * backward_pass_softmax(){
  dARRAY * softmax_outb = NULL;
  dARRAY * temp = NULL;
  dARRAY * one = NULL;
  int dims[] = {m->current_layer->DENSE->A->shape[0],m->current_layer->DENSE->A->shape[1]};
  one = ones(dims);
  temp = subtract(one,m->current_layer->DENSE->A);

  free2d(one);
  one=NULL;

  softmax_outb = multiply(m->current_layer->DENSE->A,temp);
  free2d(temp);
  temp = NULL;
  return softmax_outb;
}

Softmax * Softmax__init__(dARRAY * layer_matrix){
  Softmax * sfmax = (Softmax*)malloc(sizeof(Softmax));
  sfmax->forward = forward_pass_softmax;
  sfmax->backward = backward_pass_softmax;
  sfmax->in_dims[0] = sfmax->out_dims[0] = layer_matrix->shape[0];
  sfmax->in_dims[1] = sfmax->out_dims[1] = layer_matrix->shape[1];
  return sfmax;
}

dARRAY * (softmax)(softmax_args args){
  Softmax * sfmax = Softmax__init__(args.input);
  if(!args.status)
    return sfmax->forward();
  else
    return sfmax->backward();
  free(sfmax);
  sfmax = NULL;
}
